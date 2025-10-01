"""
End-to-End Tests for S3 Operations

This module contains integration tests for S3Operations class that interact with real AWS S3 services.
These tests require proper AWS credentials and permissions.

Note: These tests will create and delete actual S3 objects. Use with caution in production environments.
"""

import pytest
import boto3
import os
import tempfile
from typing import List
from unittest.mock import patch
import logging
import json

# Import the module under test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'lambda', 'shared'))

from s3_operations import S3Operations


class TestS3OperationsE2E:
    """
    End-to-end tests for S3Operations class.
    
    These tests interact with real AWS S3 services and require:
    1. Valid AWS credentials configured (via AWS CLI, environment variables, or IAM roles)
    2. TEST_DELETE_S3_BUCKET_FILES environment variable set to a test bucket name
    3. Appropriate S3 permissions (ListBucket, GetObject, PutObject, DeleteObject)
    
    Environment Variables:
    - TEST_DELETE_S3_BUCKET_FILES: S3 bucket name for testing (required)
    - AWS_DEFAULT_REGION: AWS region (optional, defaults to 'us-east-1')
    - AWS_ACCOUNT_ID: AWS account ID (optional, defaults to '')
    
    Example usage:
        export TEST_DELETE_S3_BUCKET_FILES=my-test-bucket
        export AWS_DEFAULT_REGION=us-west-2
        python -m pytest tests/lambda/e2e/test_s3_operations.py -v
    
    Warning: These tests create and delete real S3 objects. Use a dedicated test bucket.
    """
    
    @classmethod
    def setup_class(cls):
        """Set up test class with AWS configuration."""
        cls.region_name = os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
        cls.account_id = os.environ.get('AWS_ACCOUNT_ID', '')
        cls.test_bucket = os.environ.get('TEST_DELETE_S3_BUCKET_FILES')
        cls.access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        cls.access_token = os.environ.get('ACCESS_TOKEN')
        cls.sagemaker_bucket = os.environ.get('SAGEMAKER_BUCKET')

        print(cls.region_name)
        print(cls.account_id)
        print(cls.test_bucket)
        print(cls.access_key)
        print(cls.access_token)
        
        if not cls.test_bucket:
            pytest.skip("TEST_DELETE_S3_BUCKET_FILES environment variable not set. Skipping e2e tests.")
        
        # Create session with genai profile
        session = boto3.Session(profile_name='genai')
        
        # Initialize S3Operations instance with the session
        cls.s3_ops = S3Operations(cls.region_name, cls.account_id, session)
        
        # Initialize direct S3 client for test setup/teardown using genai profile
        cls.s3_client = session.client('s3', region_name=cls.region_name)
        
        # Print STS caller identity for debugging
        try:
            sts_client = session.client('sts', region_name=cls.region_name)
            caller_identity = sts_client.get_caller_identity()
            print(f"STS Caller Identity: {caller_identity}")
        except Exception as e:
            print(f"Failed to get STS caller identity: {e}")

        
        # Test folder prefix for isolation
        cls.test_folder_prefix = 'test-s3-operations-e2e'
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
        
        # Verify bucket exists and is accessible
        try:
            cls.s3_client.head_bucket(Bucket=cls.test_bucket)
            cls.logger.info(f"Successfully connected to test bucket: {cls.test_bucket}")
        except Exception as e:
            pytest.skip(f"Cannot access test bucket '{cls.test_bucket}': {e}")
    
    # def setup_method(self):
    #     """Set up each test method."""
    #     # Clean up any existing test files
    #     self._cleanup_test_files()
    
    # def teardown_method(self):
    #     """Clean up after each test method."""
    #     self._cleanup_test_files()
    
    def _cleanup_test_files(self):
        """Remove all test files from S3 bucket."""
        try:
            # Use the delete function we're testing to clean up
            if hasattr(self.s3_ops, 'delete_all_files_in_folder'):
                self.s3_ops.delete_all_files_in_folder(f"{self.test_bucket}/{self.test_folder_prefix}")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")
    
    def _create_test_files(self, file_paths: List[str]) -> List[str]:
        """
        Create test files in S3 bucket.
        
        Args:
            file_paths: List of file paths relative to test folder
            
        Returns:
            List of full S3 keys created
        """
        created_keys = []
        
        for file_path in file_paths:
            # Ensure file_path starts with / for consistency
            if not file_path.startswith('/'):
                file_path = '/' + file_path
                
            full_key = f"{self.test_folder_prefix}{file_path}"
            
            # Create a simple test file content with timestamp for uniqueness
            import time
            content = f"Test file content for {file_path}\nCreated at: {time.time()}"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.test_bucket,
                Key=full_key,
                Body=content.encode('utf-8'),
                ContentType='text/plain'
            )
            
            created_keys.append(full_key)
            self.logger.info(f"Created test file: s3://{self.test_bucket}/{full_key}")
        
        return created_keys
    
    def _verify_bucket_state(self, expected_count: int = None, prefix: str = None) -> List[str]:
        """
        Verify the current state of the test bucket.
        
        Args:
            expected_count: Expected number of objects (optional)
            prefix: Prefix to filter objects (optional, defaults to test_folder_prefix)
            
        Returns:
            List of current object keys
        """
        if prefix is None:
            prefix = self.test_folder_prefix
            
        result = self.s3_ops.list_objects(self.test_bucket, prefix)
        assert result['is_success'], f"Failed to list objects: {result['error_message']}"
        current_objects = result['objects']
        
        if expected_count is not None:
            assert len(current_objects) == expected_count, \
                f"Expected {expected_count} objects, found {len(current_objects)}: {current_objects}"
        
        self.logger.info(f"Bucket state verified: {len(current_objects)} objects with prefix '{prefix}'")
        return current_objects
    
    def test_list_objects_empty_bucket(self):
        """Test listing objects in empty bucket/prefix."""
        # Ensure the test folder is empty
        self._cleanup_test_files()
        
        # Verify bucket is empty
        objects = self._verify_bucket_state(expected_count=0)
        
        assert objects == []
        self.logger.info("Successfully tested empty bucket listing")
    
    def test_list_objects_single_file(self):
        """Test listing objects with a single file."""
        # Create a single test file
        test_files = ["single_test_file.txt"]
        created_keys = self._create_test_files(test_files)
        print(created_keys)
        
        # Verify and list objects
        objects = self._verify_bucket_state(expected_count=1)
        print(objects)
        
        assert objects[0] == created_keys[0]
        self.logger.info(f"Successfully listed single file: {objects[0]}")
    
    def test_list_objects_multiple_files(self):
        """Test listing objects with multiple files."""
        # Create multiple test files
        test_files = [
            "file1.txt",
            "file2.json", 
            "file3.csv",
            "subfolder/file4.txt",
            "subfolder/nested/file5.log"
        ]
        created_keys = self._create_test_files(test_files)
        
        # Verify and list objects
        objects = self._verify_bucket_state(expected_count=len(created_keys))
        
        # Verify all created files are in the result
        for key in created_keys:
            assert key in objects, f"Expected key {key} not found in {objects}"
        
        self.logger.info(f"Successfully listed {len(objects)} files")
    
    def test_list_objects_with_prefix_filter(self):
        """Test listing objects with specific prefix filtering."""
        # Create files in different subfolders
        test_files = [
            "folder1/file1.txt",
            "folder1/file2.txt", 
            "folder2/file3.txt",
            "folder2/subfolder/file4.txt",
            "other/file5.txt"
        ]
        self._create_test_files(test_files)
        
        # Verify all files were created
        self._verify_bucket_state(expected_count=5)
        
        # List objects with specific prefix
        folder1_prefix = f"{self.test_folder_prefix}/folder1/"
        result = self.s3_ops.list_objects(self.test_bucket, folder1_prefix)
        assert result['is_success'], f"Failed to list objects: {result['error_message']}"
        objects = result['objects']
        
        # Should only return files from folder1
        assert len(objects) == 2, f"Expected 2 objects with prefix {folder1_prefix}, got {len(objects)}: {objects}"
        for obj in objects:
            assert obj.startswith(folder1_prefix), f"Object {obj} doesn't start with prefix {folder1_prefix}"
        
        self.logger.info(f"Successfully filtered objects with prefix: {folder1_prefix}")
    
    def test_list_objects_directory_markers_filtered(self):
        """Test that directory markers are properly filtered out."""
        # Create files and directory markers
        test_files = [
            "file1.txt",
            "subfolder/file2.txt"
        ]
        created_keys = self._create_test_files(test_files)
        
        # Manually create directory markers (folders ending with '/')
        directory_markers = [
            f"{self.test_folder_prefix}/empty_folder/",
            f"{self.test_folder_prefix}/subfolder/"
        ]
        
        for marker in directory_markers:
            self.s3_client.put_object(
                Bucket=self.test_bucket,
                Key=marker,
                Body=b'',  # Empty content for directory marker
            )
            self.logger.info(f"Created directory marker: {marker}")
        
        # List objects using our function
        result = self.s3_ops.list_objects(self.test_bucket, self.test_folder_prefix)
        assert result['is_success'], f"Failed to list objects: {result['error_message']}"
        objects = result['objects']
        
        # Should only return actual files, not directory markers
        assert len(objects) == len(created_keys), \
            f"Expected {len(created_keys)} files, got {len(objects)}. Objects: {objects}"
        
        for obj in objects:
            assert not obj.endswith('/'), f"Directory marker {obj} was not filtered out"
            assert obj in created_keys, f"Unexpected object {obj} found"
        
        self.logger.info("Successfully filtered out directory markers")
    
    def test_list_objects_large_number_pagination(self):
        """Test listing objects with large number of files (tests pagination)."""
        # Create a large number of test files to test pagination
        # Using 25 files to keep test execution time reasonable while still testing pagination
        test_files = [f"batch_file_{i:04d}.txt" for i in range(25)]
        created_keys = self._create_test_files(test_files)
        
        # Verify and list objects
        objects = self._verify_bucket_state(expected_count=len(created_keys))
        
        # Verify all files are present
        for key in created_keys:
            assert key in objects, f"Expected key {key} not found in objects list"
        
        # Verify all objects are unique (no duplicates from pagination)
        assert len(set(objects)) == len(objects), "Found duplicate objects in pagination result"
        
        self.logger.info(f"Successfully handled pagination with {len(objects)} files")
    
    def test_list_objects_no_prefix(self):
        """Test listing objects without prefix (entire bucket scope)."""
        # Create test files in our test folder
        test_files = ["no_prefix_test.txt"]
        created_keys = self._create_test_files(test_files)
        
        # List objects without prefix (empty string) - this lists entire bucket
        result = self.s3_ops.list_objects(self.test_bucket, "")
        assert result['is_success'], f"Failed to list objects: {result['error_message']}"
        objects = result['objects']
        
        # Should include our test files plus any other files in bucket
        assert len(objects) >= len(created_keys), \
            f"Expected at least {len(created_keys)} objects, got {len(objects)}"
        
        # Verify our test files are in the result
        for key in created_keys:
            assert key in objects, f"Expected key {key} not found in bucket-wide listing"
        
        self.logger.info(f"Successfully listed {len(objects)} objects without prefix")
    
    def test_list_objects_nonexistent_bucket(self):
        """Test listing objects from non-existent bucket."""
        import uuid
        fake_bucket = f"non-existent-bucket-{uuid.uuid4().hex[:8]}"
        
        result = self.s3_ops.list_objects(fake_bucket, "")
        assert not result['is_success'], "Expected failure for non-existent bucket"
        assert result['error_code'] == 'NO_SUCH_BUCKET', f"Expected NO_SUCH_BUCKET error, got: {result['error_code']}"
        
        error_message = result['error_message'].lower()
        assert "does not exist" in error_message, \
            f"Expected bucket not found error, got: {result['error_message']}"
        
        self.logger.info("Successfully handled non-existent bucket error")

    def test_read_inference_output_file(self):
        """Test reading files on s3."""
        import uuid
        fake_s3_path = f"s3://{self.sagemaker_bucket}/parakeet-asr/output/865041f1-67a4-4161-a9bf-a8c7f1d3fa92.out"

        # Test read_inference_output_file with non-existent bucket
        result = self.s3_ops.read_inference_output_file(fake_s3_path)

        print(result)
        assert result['is_success'], "Expected failure for non-existent bucket"

        # json decode 
        content = json.loads(result['content'])
        self.logger.info(f"Decoded Content: \n {content[0]}")

    
    # @pytest.mark.skip(reason="Large batch test - skip for regular test runs")
    def test_delete_all_files_large_batch(self):
        """Test deleting a large number of files (testing batch processing)."""
        # Get bucket name from environment variables
        account_id = os.environ.get('AWS_ACCOUNT_ID', self.account_id)
        region_name = os.environ.get('AWS_DEFAULT_REGION', self.region_name)
        test_bucket = f"sagemaker-{region_name}-{account_id}"
        test_folder_prefix = "parakeet-asr/output/"

        
        # Verify files exist before deletion
        list_result = self.s3_ops.list_objects(test_bucket, test_folder_prefix)
        assert list_result['is_success'], f"Failed to list objects: {list_result['error_message']}"
        objects = list_result['objects']
        print('objects', objects)
        
        # Delete all files
        s3_path = f"s3://{test_bucket}/{test_folder_prefix}"
        delete_result = self.s3_ops.delete_all_files_in_folder(s3_path)
        assert delete_result['is_success'], f"Failed to delete files: {delete_result['error_message']}"
        deleted_count = delete_result['deleted_count']
        print('s3_path', s3_path)
        print('deleted_count', deleted_count)
        
        # Verify no files remain
        remaining_result = self.s3_ops.list_objects(test_bucket, test_folder_prefix)
        assert remaining_result['is_success'], f"Failed to list remaining objects: {remaining_result['error_message']}"
        remaining_objects = remaining_result['objects']
        assert len(remaining_objects) == 0
        
    

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])