'''
Usage:
python wii_cmpr_s3_converter.py --input myimage.png --output example.s3 --width 256 --height 256 [use-external]
still WIP, images dont display in high quality on jd dlc shops yet.
'''

import argparse
import struct
import os
import subprocess
import tempfile
from PIL import Image, ImageFilter

# -----------------------------
# Helpers
# -----------------------------
def rgb_to_565(r, g, b):
    """convert RGB888 to RGB565"""
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)

def compress_4x4_to_dxt1_smooth(pixels4x4):
    avg_r = sum(p[0] for p in pixels4x4) // 16
    avg_g = sum(p[1] for p in pixels4x4) // 16
    avg_b = sum(p[2] for p in pixels4x4) // 16


    def clamp(x): return max(0, min(255, x))
    rs = [p[0] for p in pixels4x4]
    gs = [p[1] for p in pixels4x4]
    bs = [p[2] for p in pixels4x4]

    range_r = max(rs) - min(rs)
    range_g = max(gs) - min(gs)
    range_b = max(bs) - min(bs)


    maxc = (clamp(avg_r + range_r//2), clamp(avg_g + range_g//2), clamp(avg_b + range_b//2))
    minc = (clamp(avg_r - range_r//2), clamp(avg_g - range_g//2), clamp(avg_b - range_b//2))

    c0 = rgb_to_565(*maxc)
    c1 = rgb_to_565(*minc)

    def unpack565(c):
        r = ((c >> 11) & 0x1f) << 3
        g = ((c >> 5) & 0x3f) << 2
        b = (c & 0x1f) << 3
        return (r | (r >> 5), g | (g >> 6), b | (b >> 5))

    p0 = unpack565(c0)
    p1 = unpack565(c1)

 
    palette = [p0, p1]
    if c0 > c1:
        palette.append(tuple((2*p0[i] + p1[i]) // 3 for i in range(3)))
        palette.append(tuple((p0[i] + 2*p1[i]) // 3 for i in range(3)))
    else:
        palette.append(tuple((p0[i] + p1[i]) // 2 for i in range(3)))
        palette.append((0, 0, 0))

  
    byterows = []
    for row in range(4):
        rowbyte = 0
        for col in range(4):
            px = pixels4x4[row*4 + col]
            
            px = (
                int(px[0]*0.9 + avg_r*0.1),
                int(px[1]*0.9 + avg_g*0.1),
                int(px[2]*0.9 + avg_b*0.1)
            )
            best_idx = 0
            best_dist = 1e9
            for i, c in enumerate(palette):
                dist = (px[0]-c[0])**2 + (px[1]-c[1])**2 + (px[2]-c[2])**2
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            rowbyte |= (best_idx & 0x3) << (col*2)
        byterows.append(rowbyte)

    return struct.pack('<HH', c0, c1) + bytes(byterows)

def image_to_linear_dxt1_bytes(img):
    """Convert PIL image to linear DXT1 bytes using smooth compressor (very annoying :angry:)"""
    w, h = img.size
    if w % 4 != 0 or h % 4 != 0:
        raise ValueError('Image dimensions must be multiples of 4')


    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    img = img.convert('RGB')
    pixels = img.load()

    out = bytearray()
    for by in range(0, h, 4):
        for bx in range(0, w, 4):
            block_pixels = [pixels[bx+x, by+y] for y in range(4) for x in range(4)]
            out.extend(compress_4x4_to_dxt1_smooth(block_pixels))
    return bytes(out)

def read_dds_dxt1_blocks(path):
    """extract linear DXT1 block stream from DDS"""
    with open(path, 'rb') as f:
        data = f.read()
    if data[:4] != b'DDS ':
        raise ValueError('Not a DDS file')
    return data[128:]  # skip DDS header

def swizzle_cmpr_from_linear(dxt1_bytes, width, height):
    """swizzle linear DXT1 to Wii CMPR layout"""
    blocks_x = width // 4
    blocks_y = height // 4
    if len(dxt1_bytes) != blocks_x * blocks_y * 8:
        raise ValueError('Unexpected input length')

    blocks = []
    off = 0
    for by in range(blocks_y):
        row = []
        for bx in range(blocks_x):
            row.append(dxt1_bytes[off:off+8])
            off += 8
        blocks.append(row)

    # Flip vertically so it doesnt become upside down on dlc shop
    blocks.reverse()

    macro_blocks_x = blocks_x // 2
    macro_blocks_y = blocks_y // 2
    out = bytearray()

    for my in range(macro_blocks_y):
        for mx in range(macro_blocks_x):
            base_bx = mx * 2
            base_by = my * 2
            for sx, sy in [(0,0),(1,0),(0,1),(1,1)]:
                bx = base_bx + sx
                by = base_by + sy
                block = blocks[by][bx]
                c0_le, c1_le = struct.unpack_from('<HH', block, 0)
                out.extend(struct.pack('>HH', c0_le, c1_le))
                out.extend(block[4:8])
    return bytes(out)

# -----------------------------
# Main code
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input','-i',required=True)
    parser.add_argument('--output','-o',required=True)
    parser.add_argument('--width',type=int,default=256)
    parser.add_argument('--height',type=int,default=256)
    parser.add_argument('--use-external',action='store_true',help="Use 'squish' for high-quality DXT1 compression")
    args = parser.parse_args()

    inpath = args.input
    outpath = args.output
    w, h = args.width, args.height

    if inpath.lower().endswith('.dds'):
        linear_blocks = read_dds_dxt1_blocks(inpath)
    else:
        img = Image.open(inpath).convert('RGBA')
        if img.size != (w,h):
            img = img.resize((w,h), Image.LANCZOS)

        if args.use-external:
            with tempfile.TemporaryDirectory() as td:
                tmpin = os.path.join(td, 'tmp.png')
                tmpout = os.path.join(td, 'tmp.dds')
                img.save(tmpin)
                cmd = ['squish', '--quality', 'high', '--format', 'dxt1', tmpin, tmpout]
                print('Running external compressor:', ' '.join(cmd))
                subprocess.check_call(cmd)
                linear_blocks = read_dds_dxt1_blocks(tmpout)
        else:
            print('Using smooth built-in compressor (less blocky).')
            linear_blocks = image_to_linear_dxt1_bytes(img)

    cmpr = swizzle_cmpr_from_linear(linear_blocks, w, h)
    with open(outpath, 'wb') as f:
        f.write(cmpr)
    print(f'Wrote CMPR swizzled data to {outpath}')

if __name__ == '__main__':
    main()
