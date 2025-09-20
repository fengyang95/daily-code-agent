#!/usr/bin/env python3
"""
Data Merger - Combines arXiv and WeChat data into unified format
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

class DataMerger:
    """Merges data from multiple sources while maintaining format consistency"""
    
    def __init__(self):
        self.data_dir = "data"
        self.today = datetime.now().strftime("%Y-%m-%d")
    
    def merge_daily_data(self, sources: List[str] = None) -> str:
        """Merge data from multiple sources for today with robust error handling"""
        if sources is None:
            sources = ['arxiv', 'wechat']
        
        all_articles = []
        total_loaded = 0
        
        print(f"🔄 Starting data merge for {self.today}...")
        
        for source in sources:
            try:
                articles = self._load_source_data(source)
                all_articles.extend(articles)
                total_loaded += len(articles)
                print(f"✅ Loaded {len(articles)} articles from {source}")
            except Exception as e:
                print(f"❌ Error loading data from {source}: {e}")
                continue
        
        if total_loaded == 0:
            print("⚠️  Warning: No articles found from any source!")
            print("   This might be normal if no new data is available.")
        
        # Remove duplicates based on ID
        unique_articles = self._remove_duplicates(all_articles)
        
        # Save merged data
        output_file = f"{self.data_dir}/{self.today}.jsonl"
        
        try:
            self._save_data(unique_articles, output_file)
            print(f"✅ Successfully merged {len(unique_articles)} unique articles to {output_file}")
        except Exception as e:
            print(f"❌ Error saving merged data: {e}")
            raise
        
        return output_file
    
    def _load_source_data(self, source: str) -> List[Dict[str, Any]]:
        """Load data from a specific source with robust error handling"""
        # Try different possible file patterns
        if source == 'arxiv':
            # For arXiv, also try the main data file pattern
            file_patterns = [
                f"{self.data_dir}/{self.today}.jsonl",  # Main arXiv data file
                f"{self.data_dir}/{self.today}_{source}.jsonl",
                f"{self.data_dir}/{self.today}_{source}_*.jsonl",
                f"{self.data_dir}/{source}_{self.today}.jsonl"
            ]
        else:
            file_patterns = [
                f"{self.data_dir}/{self.today}_{source}.jsonl",
                f"{self.data_dir}/{self.today}_{source}_*.jsonl",
                f"{self.data_dir}/{source}_{self.today}.jsonl"
            ]
        
        articles = []
        found_files = []
        
        for pattern in file_patterns:
            import glob
            files = glob.glob(pattern)
            
            if files:
                found_files.extend(files)
                for file_path in files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_articles = 0
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        articles.append(json.loads(line))
                                        file_articles += 1
                                    except json.JSONDecodeError as e:
                                        print(f"⚠️  Warning: Skipping invalid JSON line in {file_path}: {e}")
                                        continue
                            print(f"📄 Loaded {file_articles} articles from {file_path}")
                    except FileNotFoundError:
                        # This shouldn't happen since glob found the file, but handle gracefully
                        continue
                    except Exception as e:
                        print(f"⚠️  Warning: Error reading {file_path}: {e}")
                        continue
                break  # Stop after finding and processing the first matching pattern
        
        if not found_files:
            print(f"⚠️  Warning: No data files found for source '{source}' with patterns: {file_patterns}")
            print(f"   This is normal if no {source} data is available for {self.today}")
        
        return articles
    
    def _remove_duplicates(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on ID"""
        seen_ids = set()
        unique_articles = []
        
        for article in articles:
            article_id = article.get('id')
            if article_id and article_id not in seen_ids:
                seen_ids.add(article_id)
                unique_articles.append(article)
            elif not article_id:
                # If no ID, use title + summary as unique key
                unique_key = f"{article.get('title', '')}_{article.get('summary', '')[:50]}"
                if unique_key not in seen_ids:
                    seen_ids.add(unique_key)
                    unique_articles.append(article)
        
        return unique_articles
    
    def _save_data(self, articles: List[Dict[str, Any]], output_file: str):
        """Save merged data to file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')


def main():
    """Main function with enhanced error handling"""
    try:
        if len(sys.argv) > 1:
            date_str = sys.argv[1]
        else:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        merger = DataMerger()
        merger.today = date_str
        
        print(f"🚀 Starting data merger for date: {date_str}")
        
        # Check if arXiv data exists (optional - merger will handle missing files gracefully)
        arxiv_file = f"data/{date_str}.jsonl"
        if not os.path.exists(arxiv_file):
            print(f"ℹ️  Note: arXiv data file {arxiv_file} not found. Will proceed with WeChat data only.")
        
        # Check if WeChat data exists (optional - merger will handle missing files gracefully)
        wechat_file = f"data/{date_str}_wechat.jsonl"
        if not os.path.exists(wechat_file):
            print(f"ℹ️  Note: WeChat data file {wechat_file} not found. Will proceed with arXiv data only.")
        
        # Merge data
        merged_file = merger.merge_daily_data()
        print(f"✅ Data merging completed successfully: {merged_file}")
        
    except Exception as e:
        print(f"❌ Data merger failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
