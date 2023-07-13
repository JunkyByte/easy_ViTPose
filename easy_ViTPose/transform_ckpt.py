import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Wholebody ckpt path')
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')

    try:
        ckpt = ckpt['state_dict']
    except KeyError:
        pass

    # Remove all other heads, rename wholebody one (n.4) to keypoint_head
    patt = 'heads.4'
    for layer_name in list(ckpt.keys()):
        if patt in layer_name:
            v = ckpt.pop(layer_name)
            name = layer_name[layer_name.find(patt) + len(patt):]
            ckpt['keypoint_head' + name] = v
        elif 'associate_' in layer_name:
            ckpt.pop(layer_name)
    
    for (k, v) in ckpt.items():
        print(k, v.shape)

    torch.save({
            'state_dict': ckpt,
            }, args.ckpt.replace('.pth', '_updated.pth'))
