#!/bin/env python
# coding:utf-8

import argparse
import json
import MeCab
import unicodedata
import os

def run(indir,outfile,outdict):
    dict={}# {"単語":0,..}
    if os.path.exists(outfile):
        os.remove(outfile)

    files=os.listdir(indir)
    for file in files:
        split(indir+"/"+file,outfile,dict)

    with open(outdict,"w") as f:
        for key in dict.keys():
            f.write(str(key)+"\t"+str(dict[key])+"\n")

def split(infile,outfile,dict):
    with open(infile,"r") as f:
        js=json.load(f)
    lines=[]
    id=js["dialogue-id"]
    for turn in js["turns"]:
        lines.append(unicodedata.normalize("NFKC", turn["utterance"]))


    with open(outfile,"a") as f:
        pline=None
        for line in lines:
            line=line.replace("\n","")
            if pline is not None:
                f.write(wakati(pline,dict)+"\t"+wakati(line,dict)+"\n")
            pline=line

def wakati(s,dict):
    tagger = MeCab.Tagger("-Owakati")
    result = tagger.parse(s)
    result = result.replace("\n","").split(" ")

    def word_to_id(word):
        if word not in dict:
            dict[word]=len(dict)
        return dict[word]

    res=map(word_to_id,result)
    ret=map(str,res)
    return(" ".join(ret))

def main():
    p = argparse.ArgumentParser(description='corpus spliter')

    p.add_argument('-i','--indir', default="/Users/admin/Downloads/chat/json/init100/",help='input file')
    p.add_argument('-o','--outfile',default="/Users/admin/Downloads/chat/txt/init100.txt")
    p.add_argument('-d','--outdict',default="/Users/admin/Downloads/chat/txt/init100.dict")

    args = p.parse_args()

    run(args.indir,args.outfile,args.outdict)

if __name__ == '__main__':
    main()