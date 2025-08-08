#!/usr/bin/env python3
"""
Download MLCommons/peoples_speech dataset and upload to S3
"""

import os
# Set audio backend before any imports
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
# Disable PyTorch integration to avoid tensor errors
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

import boto3
import tempfile
import requests
from datasets import load_dataset
from tqdm import tqdm
import argparse
import soundfile as sf


def setup_s3_client():
    """Initialize S3 client"""
    return boto3.client('s3')


def upload_to_s3(s3_client, file_path, bucket_name, s3_key):
    """Upload file to S3"""
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        return True
    except Exception as e:
        print(f"Error uploading {s3_key}: {e}")
        return False


def download_and_upload_dataset(bucket_name, s3_prefix="peoples_speech", max_samples=None):
    """
    Download peoples_speech dataset and upload to S3
    
    Args:
        bucket_name: S3 bucket name
        s3_prefix: Prefix for S3 keys
        max_samples: Maximum number of samples to process (None for all)
    """
    
    # Initialize S3 client
    s3_client = setup_s3_client()
    
    # Load dataset in streaming mode with audio decoding
    print("Loading dataset...")
    from datasets import Audio
    dataset = load_dataset('MLCommons/peoples_speech', 'clean', split='train', streaming=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Process samples
    successful_uploads = 0
    failed_uploads = 0
    
    # Create progress bar
    pbar = tqdm(desc="Processing samples")
    
    for idx, sample in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break
            
        try:
            # Get audio data (already decoded by datasets)
            audio_data = sample['audio']['array']
            sample_rate = sample['audio']['sampling_rate']
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                # Write audio as WAV format
                sf.write(tmp_file.name, audio_data, sample_rate)
                
                # Create S3 key
                s3_key = f"{s3_prefix}/sample_{idx:06d}.wav"
                
                # Upload to S3
                if upload_to_s3(s3_client, tmp_file.name, bucket_name, s3_key):
                    successful_uploads += 1
                else:
                    failed_uploads += 1
                
                # Clean up temporary file
                os.unlink(tmp_file.name)
        
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            failed_uploads += 1
        
        # Update progress
        pbar.update(1)
        pbar.set_postfix({
            'Uploaded': successful_uploads,
            'Failed': failed_uploads
        })
    
    pbar.close()
    
    print(f"\nUpload complete!")
    print(f"Successful uploads: {successful_uploads}")
    print(f"Failed uploads: {failed_uploads}")
    
    return


def main():
    parser = argparse.ArgumentParser(description='Download and upload peoples_speech dataset to S3')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--prefix', default='peoples_speech', help='S3 prefix (default: peoples_speech/)')
    parser.add_argument('--max-samples', type=int, help='Maximum number of samples to process')
    parser.add_argument('--test', action='store_true', help='Test with first 10 samples only')
    
    args = parser.parse_args()
    
    if args.test:
        args.max_samples = 10
        print("Test mode: processing first 10 samples only")
    
    download_and_upload_dataset(
        bucket_name=args.bucket,
        s3_prefix=args.prefix,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()