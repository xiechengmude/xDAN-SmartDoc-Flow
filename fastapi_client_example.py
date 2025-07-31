"""
Example client for the FastAPI PDF Parser Server
Demonstrates both async and sync parsing approaches
"""
import asyncio
import httpx
import time
from pathlib import Path
from typing import Optional

class PDFParserClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0)
    
    async def parse_pdf_async(self, file_path: str, skip_merge: bool = False) -> dict:
        """
        Upload PDF for asynchronous parsing
        Returns job ID for status checking
        """
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'application/pdf')}
            params = {'skip_cross_page_merge': skip_merge}
            
            response = await self.client.post(
                f"{self.base_url}/parse",
                files=files,
                params=params
            )
            response.raise_for_status()
            return response.json()
    
    async def parse_pdf_sync(self, file_path: str, skip_merge: bool = False) -> dict:
        """
        Parse PDF synchronously (waits for completion)
        """
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'application/pdf')}
            params = {'skip_cross_page_merge': skip_merge}
            
            response = await self.client.post(
                f"{self.base_url}/parse-sync",
                files=files,
                params=params
            )
            response.raise_for_status()
            return response.json()
    
    async def get_job_status(self, job_id: str) -> dict:
        """Check the status of a parsing job"""
        response = await self.client.get(f"{self.base_url}/status/{job_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_markdown_result(self, job_id: str) -> str:
        """Get the parsed markdown text"""
        response = await self.client.get(f"{self.base_url}/result/{job_id}/markdown")
        response.raise_for_status()
        return response.text
    
    async def wait_for_completion(self, job_id: str, timeout: int = 300) -> Optional[dict]:
        """Wait for a job to complete with polling"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = await self.get_job_status(job_id)
            
            if status['status'] in ['completed', 'failed']:
                return status
            
            print(f"Job {job_id} status: {status['status']}, progress: {status.get('progress', 0)}")
            await asyncio.sleep(2)
        
        return None
    
    async def batch_parse_pdfs(self, file_paths: list, max_concurrent: int = 5):
        """Parse multiple PDFs concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def parse_with_limit(file_path):
            async with semaphore:
                return await self.parse_pdf_async(file_path)
        
        tasks = [parse_with_limit(fp) for fp in file_paths]
        return await asyncio.gather(*tasks)
    
    async def close(self):
        """Close the client connection"""
        await self.client.aclose()

# Example usage
async def main():
    # Initialize client
    client = PDFParserClient("http://localhost:8000")
    
    try:
        # Example 1: Asynchronous parsing with status checking
        print("Example 1: Async parsing with status checking")
        print("-" * 50)
        
        # Submit PDF for parsing
        job = await client.parse_pdf_async("test.pdf")
        print(f"Job submitted: {job['job_id']}")
        
        # Wait for completion
        result = await client.wait_for_completion(job['job_id'])
        if result and result['status'] == 'completed':
            print(f"Parsing completed in {result['result']['processing_time']:.2f} seconds")
            print(f"Document has {result['result']['num_pages']} pages")
            
            # Get markdown text
            markdown = await client.get_markdown_result(job['job_id'])
            print(f"Markdown length: {len(markdown)} characters")
            
            # Save to file
            with open("output.md", "w") as f:
                f.write(markdown)
        
        # Example 2: Synchronous parsing (simpler but blocks)
        print("\nExample 2: Synchronous parsing")
        print("-" * 50)
        
        result = await client.parse_pdf_sync("test.pdf")
        print(f"Pages: {result['num_pages']}")
        print(f"Markdown preview: {result['document_text'][:200]}...")
        
        # Example 3: Batch processing
        print("\nExample 3: Batch processing")
        print("-" * 50)
        
        pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        jobs = await client.batch_parse_pdfs(pdf_files)
        print(f"Submitted {len(jobs)} jobs")
        
        # Wait for all to complete
        for job in jobs:
            result = await client.wait_for_completion(job['job_id'])
            if result:
                print(f"Job {job['job_id']}: {result['status']}")
        
        # Example 4: High-performance streaming for multiple files
        print("\nExample 4: Performance test")
        print("-" * 50)
        
        start_time = time.time()
        
        # Submit 10 PDFs concurrently
        test_files = ["test.pdf"] * 10  # Replace with actual files
        jobs = await client.batch_parse_pdfs(test_files, max_concurrent=5)
        
        # Wait for all completions
        results = await asyncio.gather(*[
            client.wait_for_completion(job['job_id']) 
            for job in jobs
        ])
        
        completed = sum(1 for r in results if r and r['status'] == 'completed')
        total_time = time.time() - start_time
        
        print(f"Processed {completed}/{len(jobs)} PDFs in {total_time:.2f} seconds")
        print(f"Average: {total_time/len(jobs):.2f} seconds per PDF")
        
    finally:
        await client.close()

# Advanced example: Using with connection pooling
class HighPerformanceClient(PDFParserClient):
    def __init__(self, base_url: str = "http://localhost:8000"):
        super().__init__(base_url)
        # Use connection pooling for better performance
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
        self.client = httpx.AsyncClient(
            timeout=300.0,
            limits=limits,
            http2=True  # Enable HTTP/2 for better performance
        )

# Run the examples
if __name__ == "__main__":
    asyncio.run(main())