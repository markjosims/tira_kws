from dataloading import *

def test_tira_asr():
    tira_asr = load_tira_asr()
    for split in ['train', 'validation', 'test']:
        assert split in tira_asr
        for expected_col in ['audio', 'transcription']:
            assert expected_col in tira_asr[split].column_names

def test_get_dataloader():
    tira_asr = load_tira_asr()
    tira_asr = tira_asr['train'].select(range(300))  # use a small subset for testing
    tira_asr = prepare_dataset(tira_asr)
    dataloader = get_audio_dataloader(tira_asr)
    
    is_first = True
    assert len(dataloader) == (len(tira_asr) + BATCH_SIZE - 1)//BATCH_SIZE
    for batch in dataloader:
        if is_first:
            assert len(batch['input_features']) == BATCH_SIZE
            is_first = False
        else:
            assert len(batch['input_features']) <= BATCH_SIZE

        assert 'input_features' in batch
        assert 'label_ids' in batch

def test_sliding_window():
    ds_orig = load_tira_asr()
    ds_orig = ds_orig['train'].select(range(4))  # use a small subset for testing
    ds_sliding = prepare_dataset(ds_orig, window_size=1.0)
    assert len(ds_sliding) > len(ds_orig)
    for i in range(len(ds_sliding)):
        assert 'input_features' in ds_sliding[i]
        assert 'label_ids' in ds_sliding[i]

def test_sliding_window_clap():
    ds_orig = load_tira_asr()
    ds_orig = ds_orig['train'].select(range(4))  # use a small subset for testing
    ds_sliding = prepare_dataset(ds_orig, encoding='clap_ipa', window_size=1.0)
    assert len(ds_sliding) > len(ds_orig)
    for i in range(len(ds_sliding)):
        assert 'input_features' in ds_sliding[i]
        assert 'label_ids' in ds_sliding[i]