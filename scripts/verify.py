#!/usr/bin/env python3
"""Opt-in live text, Vision and concurrent marker smoke tests."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import struct
from urllib.request import Request, urlopen
import zlib

def image_png():
    def chunk(kind, data):
        return struct.pack('>I', len(data))+kind+data+struct.pack('>I', zlib.crc32(kind+data)&0xffffffff)
    row = b'\x00'+b'\xff\x00\x00'*64+b'\x00\x00\xff'*64
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',128,128,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(row*128))+chunk(b'IEND',b'')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8241')
    p.add_argument('--inference', action='store_true', help='Explicitly authorize new generation requests')
    a = p.parse_args()
    if not a.inference: p.error('--inference is required; this sends new generation requests')
    def chat(content):
        payload = {'model':'deepseek-v4-flash','messages':[{'role':'user','content':content}], 'temperature':0, 'max_tokens':64, 'stream':False, 'chat_template_kwargs':{'enable_thinking':False}}
        req = Request(a.url.rstrip('/')+'/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        with urlopen(req, timeout=180) as response: data = json.load(response)
        if data.get('model') != 'deepseek-v4-flash': raise ValueError('Unexpected alias')
        raw = data['choices'][0]['message']['content']
        print(json.dumps({'raw':raw}))
        return raw.removeprefix('</think>').strip()
    if chat('Reply with exactly READY.') != 'READY': raise ValueError('Text check failed')
    image = 'data:image/png;base64,'+base64.b64encode(image_png()).decode()
    answer = chat([{'type':'text','text':'Name the color on the left and the color on the right. Reply exactly: left COLOR, right COLOR.'}, {'type':'image_url','image_url':{'url':image}}]).lower().rstrip('.')
    if answer != 'left red, right blue': raise ValueError('Vision check failed: '+answer)
    markers = ['ALPHA','BRAVO','CHARLIE','DELTA']
    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(lambda marker: chat('Reply with exactly '+marker+'.'), markers))
    if answers != markers: raise ValueError('Concurrent marker check failed: '+repr(answers))
    print('PASS: text, real image, four concurrent markers. Not a quality benchmark.')

if __name__ == '__main__': main()
