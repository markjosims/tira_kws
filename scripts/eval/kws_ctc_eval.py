# YOUR CODE HERE

# imports
from tira_kws.dataloading import load_capstone_kws_cuts, get_k2_dataloader
from tira_kws.models.zipa import load_zipa_large_crctc
from tira_kws.constants import CAPSTONE_SUPERVISIONS, CAPSTONE_KEYWORDS
from lhotse import SupervisionSet
import pandas as pd
import torch

import sentencepiece
from tira_kws.constants import ZIPA_SENTENCEPIECE_MODEL

from torch.nn.functional import log_softmax
from torch.nn import CTCLoss
from torch.nn.functional import sigmoid

from wctc import wctc_loss



def get_keyword_cutset(keyword, cuts):

    negative_cuts = cuts.filter(
        lambda cut: cut.custom.get('record_type', None) == 'negative'
    )
    negative_cuts = negative_cuts.to_eager()
    
    # get cuts for the keyword
    keyword_cuts = cuts.filter(
        lambda cut: cut.custom.get('keyword', None) == keyword
    )
    keyword_cuts = keyword_cuts.to_eager()

    # concatenate with negative cuts
    kw_cutset = keyword_cuts + negative_cuts

    return kw_cutset


def get_all_kw_cutsets(keyword_list, cuts):
    kw_cutsets = {}

    for kw in keyword_list:
        kw_cutset = get_keyword_cutset(kw, cuts)
        kw_cutsets[kw] = kw_cutset

    return kw_cutsets



def main(): 

    # get keyword list
    df = pd.read_csv(CAPSTONE_KEYWORDS)
    keywords = df['keyword'].tolist()
    
    # get lhotse cuts
    cuts = load_capstone_kws_cuts()
    cuts = cuts.trim_to_supervisions()
    cuts = cuts.to_eager()
    
    kw_cutsets = get_all_kw_cutsets(keywords, cuts)

    
    # load zipa small
    model = load_zipa_large_crctc()
    model = model.to('cuda')
    

    losses = []
    sentence_infos = []        

    # make a dataframe for each batch
    batch_df_list = []

    # tokenize the keyword
    tokenizer = sentencepiece.SentencePieceProcessor()
    tokenizer.load(str(ZIPA_SENTENCEPIECE_MODEL))  


    # iterate through each of the keyword cutsets
    for kw in kw_cutsets:
        print(kw, kw_cutsets[kw])
        kw_cutset = kw_cutsets[kw]

        
        kw_tokenized = tokenizer.encode(kw)
        print(kw_tokenized)
    
        # batch each of the cutsets
        dataloader = get_k2_dataloader(kw_cutset)
        
        for batch in dataloader:
    
            # feed each batch into the model and get the ctc logits
            # this will get us the ctc logits for all the batches
            with torch.no_grad():
                inputs = batch['inputs'].to('cuda')
                input_lengths = batch['supervisions']['num_frames'].to('cuda')
                embeds, output_lengths = model.forward_encoder(inputs, input_lengths)
                ctc_logits = model.ctc_output(embeds)
                
                print('CTC LOGITS LEN: ', len(ctc_logits))

            token_indices = torch.argmax(ctc_logits, 2).tolist()
            pred_strings = tokenizer.decode(token_indices)
            #breakpoint()
            
    
            # get the probabilities from the logits for each batch
            batch_size = ctc_logits.shape[0]
            targets = torch.tensor([kw_tokenized]*batch_size)
            target_lengths = torch.tensor([len(kw_tokenized)]*batch_size)
            ctc_probs = log_softmax(ctc_logits)

            ctc_loss = CTCLoss(reduction='none')
            
            loss = ctc_loss(
                ctc_probs.permute(1, 0, 2),
                targets,
                output_lengths,
                target_lengths,
            )
            print('LOSS:', loss, loss.shape)
    
            losses.append(loss)

            losses_list = loss.tolist()

                
            wctc_loss_val = wctc_loss(
                ctc_probs.permute(1, 0, 2),
                targets,
                output_lengths,
                target_lengths,
                return_mean = False
            )
            print('WCTC LOSS: ', wctc_loss_val, wctc_loss_val.shape)

            wctc_losses_list = wctc_loss_val.tolist()
    
            
            # get sentence info for each record by batch
            batch_sent_info = []
            monocut_list = batch['supervisions']['cut']
            
            for record in monocut_list:
                record_info = {}
                #sentence = record.custom['fst_text']
                record_info['sentence'] = record.custom['fst_text']
                record_info['record_type'] = record.custom['record_type']
                record_info['keyword'] = kw
                '''
                if 'keyword' in record.custom:
                    record_info['keyword'] = record.custom['keyword']
                else:
                    record_info['keyword'] = 0'''
                
                batch_sent_info.append(record_info)
            print('BATCH SENT INFO: ', batch_sent_info)
            print()
    
            sentence_infos.append(batch_sent_info)

            # make a dataframe for each batch
            batch_df = pd.DataFrame(batch_sent_info)
            batch_df['loss'] = losses_list
            batch_df['wctc_loss'] = wctc_losses_list
            batch_df['pred_strings'] = pred_strings

            batch_df_list.append(batch_df)


    # concatenate all batch dataframes into a big df
    tot_df = pd.concat(batch_df_list)
    tot_df.to_csv('large_tot_df.csv')
            
    
            #probs = sigmoid(loss)
            #print('PROBS: ', probs)


if __name__ == "__main__":
    main()
