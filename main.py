import argparse
import torch
import random
from test import test
from train import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content', type=str,help='File path to the content image')
    parser.add_argument('--style', type=str,help='File path to the style image, or multiple style')
    parser.add_argument('--content_dir', default='E:/NHS/Data/cnt', type=str,help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', default='E:/NHS/Data/sty', type=str,help='Directory path to a batch of style images')
    parser.add_argument('--content_test_dir', default='./data/cnt', type=str)
    parser.add_argument('--style_test_dir', default='./data/sty', type=str)
    parser.add_argument('--vgg', type=str, default='E:/NHS/Weight/vgg_normalised.pth')
    parser.add_argument('--checkpoints_dir', default='./experiments',help='Directory to save the model')
    parser.add_argument('--results_dir', default='./experiments',help='Directory to save the results')
    parser.add_argument('--log_dir', default='./logs',help='Directory to save the log')
    parser.add_argument('--decoder_path', type=str, default='./experiments/decoder_iter_160000.pth')
    parser.add_argument('--mamba_path', type=str, default='./experiments/mamba_iter_160000.pth')
    parser.add_argument('--embedding_path', type=str, default='./experiments/embedding_iter_160000.pth')
    parser.add_argument('--continue_train', action='store_true') 
    parser.add_argument('--resume_iter', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay', type=float, default=1e-3)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--print_every', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--content_weight', type=float, default=7.0)
    parser.add_argument('--l1_weight', type=float, default=70.0)
    parser.add_argument('--l2_weight', type=float, default=1.0)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=10000)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--hidden_dim', default=512, type=int,help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--seed', default=777, type=int,help="Seed for reproducibility")
    parser.add_argument('--model_name',type=str ,default='TMamba')
    parser.add_argument('--use_pos_embed', action='store_true') 
    parser.add_argument('--rnd_style', action='store_true') 
    parser.add_argument('--output_dir', type=str, default='./output',help='Directory to save the output image(s)')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--mode', type=str,default='train',choices=['train' , 'test'])
    parser.add_argument('--d_state', type=int, default=16, help='Mamba hidden state dimension')
    parser.add_argument('--self_attn_layers', type=int, default=3, help='Self-Attention Layer')
    parser.add_argument('--cross_attn_layers', type=int, default=3, help='Cross-Attention Layer')
    parser.add_argument('--attn_heads', type=int, default=8, help='Number of attention heads in attention modules')
    parser.add_argument('--attn_dropout', type=float, default=0.1, help='Dropout rate in attention modules')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    if args.mode == 'train':
        print(args)
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        print("Specify valid mode")