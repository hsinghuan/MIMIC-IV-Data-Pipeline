#!/usr/bin/env python
# coding: utf-8

import pickle
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import argparse
import sys

from model.model_utils import init

import warnings
# Add this near the top of your script
warnings.filterwarnings("ignore", message="To copy construct from a tensor")

def getXY(ids, labels, data_icu=True, diag_flag=True, proc_flag=True, out_flag=True, chart_flag=True, med_flag=True, lab_flag=True):
    """
    Extract features (X) and labels (Y) from the data for deep learning models.
    This function follows the same logic as in dl_train.py
    """
    # Initialize vocabulary sizes and mappings
    cond_vocab_size, proc_vocab_size, med_vocab_size, out_vocab_size, chart_vocab_size, lab_vocab_size, eth_vocab, gender_vocab, age_vocab, ins_vocab = init(
        diag_flag, proc_flag, out_flag, chart_flag, med_flag, lab_flag)
    
    dyn_df = []
    meds = torch.zeros(size=(0,0))
    chart = torch.zeros(size=(0,0))
    proc = torch.zeros(size=(0,0))
    out = torch.zeros(size=(0,0))
    lab = torch.zeros(size=(0,0))
    stat_df = torch.zeros(size=(1,0))
    demo_df = torch.zeros(size=(1,0))
    y_df = []
    keys_to_cols = {}
    
    # Get the keys (modalities) from the first sample
    dyn = pd.read_csv('./data/csv/'+str(ids[0])+'/dynamic.csv', header=[0,1])
    keys = dyn.columns.levels[0]
    
    # Initialize dynamic data arrays
    for i in range(len(keys)):
        dyn_df.append(torch.zeros(size=(1,0)))
    
    print('Processing samples:', len(ids))
    for i, sample in enumerate(tqdm(ids, desc="Processing")):
        if data_icu:
            y = labels[labels['stay_id']==sample]['label']
        else:
            y = labels[labels['hadm_id']==sample]['label']
        y_df.append(int(y))
        
        # Read dynamic data
        dyn = pd.read_csv('./data/csv/'+str(sample)+'/dynamic.csv', header=[0,1])
        
        # Process each modality
        for key in range(len(keys)):
            dyn_temp = dyn[keys[key]]
            # record the column names for each modality, (only for the first sample)
            if i == 0:
                keys_to_cols[keys[key]] = dyn_temp.columns.tolist()
            dyn_temp = dyn_temp.to_numpy()
            dyn_temp = torch.tensor(dyn_temp)
            dyn_temp = dyn_temp.unsqueeze(0)
            dyn_temp = torch.tensor(dyn_temp)
            dyn_temp = dyn_temp.type(torch.LongTensor)
            
            if dyn_df[key].nelement():
                dyn_df[key] = torch.cat((dyn_df[key], dyn_temp), 0)
            else:
                dyn_df[key] = dyn_temp
        
        # Read static data
        stat = pd.read_csv('./data/csv/'+str(sample)+'/static.csv', header=[0,1])
        stat = stat['COND']
        stat = stat.to_numpy()
        stat = torch.tensor(stat)
        
        if stat_df[0].nelement():
            stat_df = torch.cat((stat_df, stat), 0)
        else:
            stat_df = stat
        
        # Read demographic data
        demo = pd.read_csv('./data/csv/'+str(sample)+'/demo.csv', header=0)
        demo["gender"].replace(gender_vocab, inplace=True)
        demo["ethnicity"].replace(eth_vocab, inplace=True)
        demo["insurance"].replace(ins_vocab, inplace=True)
        demo["Age"].replace(age_vocab, inplace=True)
        demo = demo[["gender", "ethnicity", "insurance", "Age"]]
        demo = demo.values
        demo = torch.tensor(demo)
        
        if demo_df[0].nelement():
            demo_df = torch.cat((demo_df, demo), 0)
        else:
            demo_df = demo
    
    # Assign each modality to its variable
    for k in range(len(keys)):
        if keys[k] == 'MEDS':
            meds = dyn_df[k]
        if keys[k] == 'CHART':
            chart = dyn_df[k]
        if keys[k] == 'OUT':
            out = dyn_df[k]
        if keys[k] == 'PROC':
            proc = dyn_df[k]
        if keys[k] == 'LAB':
            lab = dyn_df[k]
    
    # Convert to appropriate tensor types
    stat_df = torch.tensor(stat_df)
    stat_df = stat_df.type(torch.LongTensor)
    
    demo_df = torch.tensor(demo_df)
    demo_df = demo_df.type(torch.LongTensor)
    
    y_df = torch.tensor(y_df)
    y_df = y_df.type(torch.LongTensor)
    
    return {
        'meds': meds,
        'chart': chart,
        'out': out,
        'proc': proc,
        'lab': lab,
        'stat': stat_df,
        'demo': demo_df,
        'y': y_df,
        'keys_to_cols': keys_to_cols
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate X and Y data for deep learning models')
    parser.add_argument('--data_icu', action='store_true', help='Whether data is ICU data')
    parser.add_argument('--output', type=str, default='./XY_dl_data.pkl', help='Output file path')
    parser.add_argument('--diag', action='store_true', help='Include diagnosis data')
    parser.add_argument('--proc', action='store_true', help='Include procedure data')
    parser.add_argument('--out', action='store_true', help='Include output data')
    parser.add_argument('--chart', action='store_true', help='Include chart data')
    parser.add_argument('--med', action='store_true', help='Include medication data')
    parser.add_argument('--lab', action='store_true', help='Include lab data')
    args = parser.parse_args()
    
    # Set flags for different modalities
    diag_flag = args.diag
    proc_flag = args.proc
    out_flag = args.out
    chart_flag = args.chart
    med_flag = args.med
    lab_flag = args.lab
    
    print(f"diag_flag: {diag_flag}, proc_flag: {proc_flag}, out_flag: {out_flag}, chart_flag: {chart_flag}, med_flag: {med_flag}, lab_flag: {lab_flag}")

    # If no flags are specified, use all modalities
    if not any([diag_flag, proc_flag, out_flag, chart_flag, med_flag, lab_flag]):
        diag_flag = proc_flag = out_flag = chart_flag = med_flag = lab_flag = True
    
    # Load labels
    labels = pd.read_csv('./data/csv/labels.csv', header=0)
    
    # Get all sample IDs
    if args.data_icu:
        ids = labels['stay_id'].tolist()
    else:
        ids = labels['hadm_id'].tolist()
    
    print(f"Total Samples: {len(ids)}")
    print(f"Positive Samples: {labels['label'].sum()}")
    
    # Extract features and labels
    data = getXY(ids, labels, args.data_icu, diag_flag, proc_flag, out_flag, chart_flag, med_flag, lab_flag)
    
    # Save the data
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Data saved to {args.output}")
    
    # Print summary of the data
    print("\nData Summary:")
    for key, value in data.items():
        if key == 'keys_to_cols':
            continue
        elif key != 'y':
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value.shape}, {value.sum().item()} positive")

if __name__ == "__main__":
    main()
