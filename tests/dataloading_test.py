from dataloading import *

def test_tira_asr():
    tira_asr = load_tira_asr()
    for split in ['train', 'validation', 'test']:
        assert split in tira_asr
        for expected_col in ['audio', 'transcription']:
            assert expected_col in tira_asr[split].column_names

def test_get_dataloader():
    tira_asr = load_tira_asr()
    dataloader = get_audio_dataloader(tira_asr['train'])
    
    is_first = True
    assert len(dataloader) == (len(tira_asr['train']) + BATCH_SIZE - 1)//BATCH_SIZE
    for batch in dataloader:
        if is_first:
            assert len(batch['input_features']) == BATCH_SIZE
            is_first = False
        else:
            assert len(batch['input_features']) <= BATCH_SIZE

        assert 'input_features' in batch
        assert 'label_ids' in batch