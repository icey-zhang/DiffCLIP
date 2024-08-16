
import torch
import numpy as np
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

ce_loss = nn.CrossEntropyLoss()

def text_valid(classnames):
    
    ctx_init_1 = "a patch of a"    #p1
    ctx_init_2 = "a fine grained patch of a"    #p2
    ctx_init_3 = "a multimodal fusion patch of a"    #p3
    ctx_init_4 = "a patch of a"    #p4
    classnames = [name.replace("_", " ") for name in classnames]
    prompts = [ctx_init_3 + " " + name + "." for name in classnames]
    tokenized_prompts_fuse = torch.cat([tokenize(p) for p in prompts])  # (n_cls, n_tkn)


    return tokenized_prompts_fuse
def tr_acc(model, image, image_LIDAR, label, diffusion,classnames):
    train_dataset = TensorDataset(torch.tensor(image), torch.tensor(image_LIDAR), torch.tensor(label))
    train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = False)
    train_loss = 0
    corr_num = 0
    for idx, (image_batch, image_LIDAR_batch, label_batch) in enumerate(train_loader):
        trans_image_batch = image_batch.cuda()
        trans_image_LIDAR_batch = image_LIDAR_batch.cuda()
        label_batch = label_batch.cuda()
        t = torch.randint(0, diffusion.num_timesteps, (trans_image_batch.shape[0],)).cuda()
        text = text_valid(classnames).cuda()
        logits,_ = model(trans_image_batch, trans_image_LIDAR_batch, t,text)

        if isinstance(logits,tuple):
            logits = logits[-1]
        pred = torch.max(logits, dim=1)[1]
        loss = ce_loss(logits, label_batch)                
        train_loss = train_loss + loss.cpu().data.numpy()
        corr_num = torch.eq(pred, label_batch).float().sum().cpu().numpy() + corr_num   
    return corr_num/image.shape[0], train_loss/(idx+1)

from simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from pkg_resources import packaging
def tokenize(texts, context_length: int = 77, truncate: bool = False):
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def text_target(classnames,batch_target):
    
    ctx_init_1 = "a patch of a"    #p1
    ctx_init_2 = "a fine grained patch of a"    #p2
    ctx_init_3 = "a multimodal fusion patch of a"    #p3
    ctx_init_4 = "a patch of a"    #p4
    classnames = [name.replace("_", " ") for name in classnames]
    prompts = [ctx_init_3 + " " + classnames[batch_target[i].item()] + "." for i in range(len(batch_target))]
    tokenized_prompts_fuse = torch.cat([tokenize(p) for p in prompts])  # (n_cls, n_tkn)


    return tokenized_prompts_fuse



def pre_train(model, train_image, train_image_LIDAR, train_label,validation_image, validation_image_LIDAR, validation_label, epoch, optimizer, scheduler, bs, diffusion,classnames, val = False, ):
    train_dataset = TensorDataset(torch.tensor(train_image), torch.tensor(train_image_LIDAR), torch.tensor(train_label))
    train_loader = DataLoader(dataset = train_dataset, batch_size = bs, shuffle = True)
    Train_loss = []
    Train_acc = []
    Val_loss = []
    Val_acc = []
    BestAcc = 0
    for i in range(epoch): 
        model.train()
        train_loss = 0
        for idx, (image_batch, image_LIDAR_batch, label_batch) in enumerate(train_loader):

            trans_image_batch = image_batch.cuda()
            trans_image_LIDAR_batch = image_LIDAR_batch.cuda()
            label_batch = label_batch.cuda()
            text = text_target(classnames,label_batch).cuda()
            t = torch.randint(0, diffusion.num_timesteps, (trans_image_batch.shape[0],)).cuda()
            # logits = model(trans_image_batch, trans_image_LIDAR_batch, t,text)
            # loss = ce_loss(logits, label_batch)
            logits_per_image_x,logits_per_text = model(trans_image_batch, trans_image_LIDAR_batch, t,text)
            labels = torch.arange(len(logits_per_image_x)).to(logits_per_image_x.device)
            loss = ce_loss(logits_per_image_x, labels)
            loss += ce_loss(logits_per_text, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss = train_loss + loss
        scheduler.step()
        train_loss = train_loss / (idx + 1)
        train_acc, tr_loss = tr_acc(model.eval(), train_image, train_image_LIDAR, train_label, diffusion,classnames)
        val_acc, val_loss= tr_acc(model.eval(), validation_image, validation_image_LIDAR, validation_label, diffusion,classnames)
        if val_acc > BestAcc:
            torch.save(model.state_dict(), 'Best_val_model/' + 'net_params.pkl')
            BestAcc = val_acc
        print("epoch {}, training loss: {:.4f}, train acc:{:.4f}, valid acc:{:.4f}".format(i, train_loss.item(), train_acc*100, val_acc*100))

        if val:
            Train_loss.append(tr_loss)
            Val_loss.append(val_loss) 
            Train_acc.append(train_acc)
            Val_acc.append(val_acc)
    if val:
        return model,  [Train_loss, Train_acc, Val_loss, Val_acc]
    else:                  
        return model

def test_batch(model, image, image_LIDAR, index, BATCH_SIZE,  nTrain_perClass, nvalid_perClass, halfsize, diffusion,classnames):
    ind = index[0][nTrain_perClass[0]+ nvalid_perClass[0]:,:]
    nclass = len(index)
    true_label = np.zeros(ind.shape[0], dtype = np.int32)
    for i in range(1, nclass):
        ddd = index[i][nTrain_perClass[i] + nvalid_perClass[i]:,:]
        ind = np.concatenate((ind, ddd), axis = 0)
        tr_label = np.ones(ddd.shape[0], dtype = np.int32) * i
        true_label = np.concatenate((true_label, tr_label), axis = 0)
    test_index = np.copy(ind)
    length = ind.shape[0]
    if length % BATCH_SIZE != 0:
        add_num = BATCH_SIZE - length % BATCH_SIZE        
        ff = range(length)    
        add_ind = np.random.choice(ff, add_num, replace = False)
        add_ind = ind[add_ind]        
        ind = np.concatenate((ind,add_ind), axis =0)

    pred_array = np.zeros([ind.shape[0],nclass], dtype = np.float32)
    n = ind.shape[0] // BATCH_SIZE
    windowsize = 2 * halfsize + 1
    image_batch = np.zeros([BATCH_SIZE, windowsize, windowsize, image.shape[2]], dtype=np.float32)
    image_LIDAR_batch = np.zeros([BATCH_SIZE, windowsize, windowsize, image_LIDAR.shape[2]], dtype=np.float32)
    for i in range(n):
        for j in range(BATCH_SIZE):
            m = ind[BATCH_SIZE*i+j, :]
            image_batch[j,:,:,:] = image[(m[0] - halfsize):(m[0] + halfsize + 1),
                                                   (m[1] - halfsize):(m[1] + halfsize + 1),:]
            image_b = np.transpose(image_batch,(0,3,1,2))
            image_LIDAR_batch[j,:,:,:] = image_LIDAR[(m[0] - halfsize):(m[0] + halfsize + 1),
                                                   (m[1] - halfsize):(m[1] + halfsize + 1),:]
            image_LIDAR_b = np.transpose(image_LIDAR_batch,(0,3,1,2))
            
        t = torch.randint(0, diffusion.num_timesteps, (image_b.shape[0],)).cuda()
        text = text_valid(classnames).cuda()
        logits,_ = model(torch.tensor(image_b).cuda(), torch.tensor(image_LIDAR_b).cuda(), t,text)
        if isinstance(logits,tuple):
            logits = logits[-1]            
        pred_array[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = torch.softmax(logits, dim = 1).cpu().data.numpy()
    pred_array = pred_array[range(length)]
    predict_label  = np.argmax(pred_array, axis=1)
    
    
    confusion_matrix = metrics.confusion_matrix(true_label, predict_label)
    overall_accuracy = metrics.accuracy_score(true_label, predict_label)
    
    true_cla = np.zeros(nclass,  dtype=np.int64)
    for i in range(nclass):
        true_cla[i] = confusion_matrix[i,i]
    test_num_class = np.sum(confusion_matrix,1)
    test_num = np.sum(test_num_class)
    num1 = np.sum(confusion_matrix,0)
    po = overall_accuracy
    pe = np.sum(test_num_class*num1)/(test_num*test_num)
    kappa = (po-pe)/(1-pe)*100
    true_cla = np.true_divide(true_cla,test_num_class)*100 
    average_accuracy = np.average(true_cla)
    print('overall_accuracy: {0:f}'.format(overall_accuracy*100)) 
    print('average_accuracy: {0:f}'.format(average_accuracy))  
    print('kappa:{0:f}'.format(kappa))
    return true_cla, overall_accuracy*100, average_accuracy, kappa, true_label, predict_label, test_index, confusion_matrix, pred_array
