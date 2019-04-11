import torch
from tqdm import tqdm
import argparse
from metal.mmtl.cxr.cxr_tasks import create_cxr_datasets, create_cxr_dataloaders

import logging

# Configuring logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Before running, must set CXRDATA env to appropriate location
if "__name__" == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--dataset_name', '-lp', required=True,
                   type=str, help='name of dataset' )
    parser.add_argument('--slice_module_path','-smp', type=str,
                   help='path to saved slice module')
    parser.add_argument('--slice_name', '-sn', type=str,
                   required=True, help='name of slice')
    parser.add_argument('--slice_label', '-sl', type=int, required=True,
                   help='module label associated with slice, assume softmax')
    parser.add_argument('--load_type', '-st', type=str,
                   default='torch', choices=['torch','pickle'])
    parser.add_argument('--save_dir', '-sd', type=int, required=True,
                   help='path to save slice labels')

    # detecting GPU
    use_cuda = torch.cuda.is_available()

    # Getting slice labels on all subsets
    splits = ["train", "dev", "test"]

    logger.info(f"Getting datasets and dataloaders...")
    # Getting all datasets
    datasets = create_cxr_datasets(
                dataset_name=args.dataset_name,
                splits=splits,
                subsample=None,
                pooled=False,
                finding="ALL",
                verbose=True,
                dataset_kwargs={'num_workers':0, 'batch_size':16}
                get_uid = True
                )

    # Wrap datasets with DataLoader objects
    data_loaders = create_cxr_dataloaders(
            datasets,
            dl_kwargs=dl_kwargs,
            split_prop=None,
            splits=splits,
            seed=123,
        )

    logger.info(f"Loading saved slice module...")
    # Loading saved slice module; .forward() gives slice label
    if args.save_type == 'torch':
        slice_module = torch.load(args.slice_module_path)
    elif args.save_type == 'pickle':
        with open(args.slice_module_path, 'rb') as fl:
            slice_module = pickle.load(fl)
    else:
        raise ValueError('Unrecognized file type, must be pickle or torch')

   # Creating output directory and saving
   if not os.path.exists(args.save_dir):
       os.mkdir(args.save_dir)

    # Setting cuda if avialable
    if use_cuda:
        slice_module.cuda()

    # Evaluating slice module
    slice_label_dict = {}
    for split in splits:
        logger.info(f"Evaluating {split} split...")
        for ii, batch in tqdm(enumerate(data_loaders[split])):
            X, y, uid = batch
            X = Variable(X)
            if use_cuda:
                X = X.cuda()
            slice_preds = slice_module(X)
            slice_labels = torch.argmax(slice_labels)
            batch_dict = dict(zip((uid, slice_labels.cpu().data.numpy()))
            slice_label_dict.update(batch_dict)

        # Creating dataframe
        slice_df = pd.DataFrame(
                        {'data_index' : slice_label_dict.keys(),
                         'slice_label' : slice_label_dict.values()
                        })

        slice_df.to_csv(f"{args.save_dir}/{split}.tsv",sep='\t','index'=False)
