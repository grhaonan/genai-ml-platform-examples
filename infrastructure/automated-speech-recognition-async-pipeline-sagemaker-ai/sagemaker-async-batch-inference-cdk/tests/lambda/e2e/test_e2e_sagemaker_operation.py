"""
Unit tests for S3SageMakerProcessor class.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import os
import sys

# Add the lambda directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../lambda/s3-sagemaker-processor'))

from index import S3SageMakerProcessor, lambda_handler
from config import ConfigManager
from event_validator import EventValidator
from sagemaker_operations import SageMakerOperations


class TestLambdaHandler(unittest.TestCase):
    """Test cases for lambda_handler function."""
    
    @patch('index.processor')
    def test_sagemaker_async_endpoint(self, mock_processor):
        # call submit_file_for_inference without mocking

        # Initialize components
        config_manager = ConfigManager()
        event_validator = EventValidator()

        sagemakerOperation = SageMakerOperations(
            region_name=ConfigManager.AWS_REGION,
            account_id=ConfigManager.AWS_ACCOUNT_ID
        )

        sagemaker_bucket = os.environ.get('SAGEMAKER_BUCKET', '')
        result = sagemakerOperation.submit_file_for_inference(
            endpoint_name="parakeet-async-endpoint-1753856615",
            content_type="audio/x-audio",
            input_location=f"s3://{sagemaker_bucket}/parakeet-asr/data/test.wav",
        )

        print('testingsssssss')
        print(result)

    @patch('index.processor')
    def test_for_loop_async(self, mock_processor):
        # Initialize components
        config_manager = ConfigManager()
        event_validator = EventValidator()

        sagemakerOperation = SageMakerOperations(
            region_name=ConfigManager.AWS_REGION,
            account_id=ConfigManager.AWS_ACCOUNT_ID
        )

        import time
        start_time = time.time()
        
        sagemaker_bucket = os.environ.get('SAGEMAKER_BUCKET', '')
        results = []
        for i in range(10):
            loop_start = time.time()
            result = sagemakerOperation.submit_file_for_inference(
                endpoint_name="parakeet-async-endpoint-1753856615",
                content_type="audio/x-audio",
                input_location=f"s3://{sagemaker_bucket}/parakeet-asr/data/test.wav",
            )
            loop_end = time.time()
            results.append(result)
            print(f"Loop {i+1}: {loop_end - loop_start:.2f} seconds, Result: {result}")
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 10
        
        print(f"\nTotal time for 10 submissions: {total_time:.2f} seconds")
        print(f"Average time per submission: {avg_time:.2f} seconds")
        
        print('Sequential loop test completed')
        print(f"All results: {results}")

    @patch('index.processor')
    def test_batch_send_async(self, mock_processor):
        # Send 10 concurrent requests to SageMaker async endpoint using batch function
        import time

        # Initialize components
        config_manager = ConfigManager()
        event_validator = EventValidator()

        sagemaker_operation = SageMakerOperations(
            region_name=ConfigManager.AWS_REGION,
            account_id=ConfigManager.AWS_ACCOUNT_ID
        )

        # Prepare list of S3 URLs (using same URL for testing)
        sagemaker_bucket = os.environ.get('SAGEMAKER_BUCKET', '')
        s3_urls = [
            f"s3://{sagemaker_bucket}/parakeet-asr/data/test.wav"
        ] * 10  # Create 10 identical URLs for testing

        start_time = time.time()
        
        # Use the new batch submit function
        results = sagemaker_operation.batch_submit_files_for_inference(
            endpoint_name="parakeet-async-endpoint-1753856615",
            s3_urls=s3_urls,
            batch_count=10,
            content_type="audio/x-audio"
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(results)
        
        # Print summary
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        print(f"\n=== Batch Request Summary ===")
        print(f"Total requests: {len(results)}")
        print(f"Successful: {len(successful_requests)}")
        print(f"Failed: {len(failed_requests)}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per request: {total_time/len(results):.2f} seconds")
        
        if failed_requests:
            print(f"\nFailed requests:")
            for failed in failed_requests:
                print(f"  Request {failed['request_id']}: {failed['error']}")
        
        # Assert that at least some requests succeeded
        self.assertGreater(len(successful_requests), 0, "At least one request should succeed")
        
        # Verify all results have the expected structure
        for result in results:
            self.assertIn('request_id', result)
            self.assertIn('success', result)
            self.assertIn('s3_url', result)
            if result['success']:
                self.assertIn('result', result)
            else:
                self.assertIn('error', result)

    @patch('index.processor')
    def test_batch_send_async_with_different_urls(self, mock_processor):
        # Test batch submission with different S3 URLs
        import time

        # Initialize components
        config_manager = ConfigManager()
        event_validator = EventValidator()

        sagemaker_operation = SageMakerOperations(
            region_name=ConfigManager.AWS_REGION,
            account_id=ConfigManager.AWS_ACCOUNT_ID
        )

        # Prepare list of different S3 URLs
        sagemaker_bucket = os.environ.get('SAGEMAKER_BUCKET', '')
        s3_urls = [
            f"s3://{sagemaker_bucket}/parakeet-asr/data/test.wav",
            f"s3://{sagemaker_bucket}/parakeet-asr/data/test.wav",
            f"s3://{sagemaker_bucket}/parakeet-asr/data/test.wav",  # Duplicate for testing
        ]

        start_time = time.time()
        
        # Use the new batch submit function with smaller batch count
        results = sagemaker_operation.batch_submit_files_for_inference(
            endpoint_name="parakeet-async-endpoint-1753856615",
            s3_urls=s3_urls,
            batch_count=3,
            content_type="audio/x-audio"
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n=== Different URLs Batch Test Results ===")
        for result in results:
            print(f"Request {result['request_id']}: {result['success']} - {result['s3_url']}")
            if not result['success']:
                print(f"  Error: {result['error']}")
        
        # Print summary
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        print(f"\nTotal requests: {len(results)}")
        print(f"Successful: {len(successful_requests)}")
        print(f"Failed: {len(failed_requests)}")
        print(f"Total time: {total_time:.2f} seconds")
        
        # Verify we got results for all URLs
        self.assertEqual(len(results), len(s3_urls), "Should get result for each S3 URL")
        
        # Verify results are ordered by request_id
        request_ids = [r['request_id'] for r in results]
        self.assertEqual(request_ids, sorted(request_ids), "Results should be ordered by request_id")

    @patch('index.processor')
    def test_batch_send_async_error_handling(self, mock_processor):
        # Test batch submission error handling
        config_manager = ConfigManager()
        event_validator = EventValidator()

        sagemaker_operation = SageMakerOperations(
            region_name=ConfigManager.AWS_REGION,
            account_id=ConfigManager.AWS_ACCOUNT_ID
        )

        # Test with empty list
        with self.assertRaises(ValueError) as context:
            sagemaker_operation.batch_submit_files_for_inference(
                endpoint_name="test-endpoint",
                s3_urls=[],
                batch_count=5
            )
        self.assertIn("s3_urls list cannot be empty", str(context.exception))

        # Test with invalid batch_count
        with self.assertRaises(ValueError) as context:
            sagemaker_operation.batch_submit_files_for_inference(
                endpoint_name="test-endpoint",
                s3_urls=["s3://test/file.wav"],
                batch_count=0
            )
        self.assertIn("batch_count must be greater than 0", str(context.exception))

        print("Error handling tests passed")
        

    @patch('index.processor')
    def test_sagemaker_async_check_result(self, mock_processor):
        # call submit_file_for_inference without mocking

        # Initialize components
        config_manager = ConfigManager()
        event_validator = EventValidator()

        sagemakerOperation = SageMakerOperations(
            region_name=ConfigManager.AWS_REGION,
            account_id=ConfigManager.AWS_ACCOUNT_ID
        )

        result = sagemakerOperation.submit_file_for_inference(
            endpoint_name="parakeet-async-endpoint-1753856615",
            content_type="application/json",
            input_location="s3://sagemaker-async-input/20250730/ElevenLabs_2025-07-25T23_10_15_Liam_pre_sp100_s50_sb75_v3.mp3",
        )


        
        print('testingsssssss')
        print(result)
      


if __name__ == '__main__':
    unittest.main()