#!/bin/env python
# coding:utf-8

import argparse
import json
import MeCab
import unicodedata
import os

def run(indir,indic,outfile,outdict,mode,append):
    dict=load_dict(indic)
    if len(dict.keys())<=0:
        dict={"<eos>":0,"<unk>":1}# {"単語":0,..}  </s>:終端記号, <unk>:未知語

    if append is True and os.path.exists(outfile):
        os.remove(outfile)

    files=os.listdir(indir)
    for file in files:
        if mode == "chat":
            chat_analyze(indir+"/"+file,outfile,dict)
        elif mode =="nuc":
            nuc_analyze(indir+"/"+file,outfile,dict)

    with open(outdict,"w") as f:
        for key in dict.keys():
            f.write(str(key)+"\t"+str(dict[key])+"\n")

# TODO
def nuc_analyze(infile,outfile,dict):
    with open(infile,"r") as f:
        sentence=""
        for line in f.readlines():
            if line.find("＠")==0 or line.find("％")==0:
                continue
            if line.index("：")>=0:
                sentence=line.split("：")[1]
            else:
                sentence=sentence+line
    pass

def chat_analyze(infile,outfile,dict):
    if infile.find("._")>=0:
        return
    with open(infile,"r") as f:
        js=json.load(f)
    lines=[]
    #id=js["dialogue-id"]
    for turn in js["turns"]:
        lines.append(unicodedata.normalize("NFKC", turn["utterance"]))

    with open(outfile,"a") as f:
        pline=None
        for line in lines:
            line=line.replace("\n","")
            if pline is not None and line is not None and len(pline)>0 and len(line)>0:
                f.write(wakati(pline,dict,True,True)+"\t"+wakati(line,dict,False,True)+"\n")
            pline=line

def load_dict(indict):

    dict={}
    if os.path.exists(indict) is False:
        return(dict)

    # 辞書の読み込み
    with open(indict,"r") as f:
        for line in f.readlines():
            items=line.replace("\n","").split("\t")
            dict[items[0]]=int(items[1])
    return(dict)

def wakati_list(s,dict,is_encode,is_train):
    tagger = MeCab.Tagger("-Owakati")
    s = s.replace("\n","")
    result = tagger.parse(s)
    result = result.replace("\n","").split(" ")
    result.remove("")

    def word_to_id(word):
        if is_train:
            if word not in dict:
                dict[word]=len(dict)
            return dict[word]
        else:
            if word not in dict:
                return dict["<unk>"]
            return dict[word]
    # http://hiroto1979.hatenablog.jp/entry/2016/02/10/112352
    res = [word_to_id(i) for i in result]

    #if is_encode:
    #    res.reverse()
    #else:
    #    res.append(dict["<eos>"])  # 終端記号
    return(res)

def wakati(s,dict,is_encode,is_train):
    res=wakati_list(s,dict,is_encode,is_train)
    ret=map(str,res)
    s=" ".join(ret)
    return(s)

def main():
    p = argparse.ArgumentParser(description='corpus spliter')

    p.add_argument('--indir', default="/Volumes/DATA/data/chat/json/init100/",help='input file')
    #p.add_argument('--indir', default="/Volumes/DATA/data/chat/json/rest1046/",help='input file')
    p.add_argument('--append', default=True, help="append outfile if exists")
    p.add_argument('--mode', default="chat", help="chat or nuc")
    p.add_argument('--indict',default="/Volumes/DATA/data/chat/txt/init.dict")
    p.add_argument('--outfile',default="/Volumes/DATA/data/chat/txt/init.txt")
    p.add_argument('--outdict',default="/Volumes/DATA/data/chat/txt/init.dict")
    args = p.parse_args()

    run(args.indir,args.indict,args.outfile,args.outdict,args.mode,args.append)

if __name__ == '__main__':
    main()