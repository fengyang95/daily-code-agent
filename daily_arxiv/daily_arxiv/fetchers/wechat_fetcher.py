#!/usr/bin/env python3
"""
WeChat Data Fetcher
Fetches articles from WeChat sources using Unified Search API and converts them to arXiv format
"""

import json
import os
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Any
import requests


class WeChatFetcher:
    """Fetches and processes WeChat articles using Unified Search API to match arXiv format"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # WeChat Search API configuration - updated endpoint
        self.api_base_url = self.config.get('api_base_url', 'http://47.117.133.51:30015')
        self.api_endpoint = '/api/weixin/search/v1'  # New endpoint
        self.api_token = self.config.get('api_token', os.environ.get('WECHAT_API_TOKEN', ''))

        # Retry configuration
        self.max_retries = self.config.get('max_retries', 5)
        self.retry_base_delay = self.config.get('retry_base_delay', 1)  # Base delay in seconds
        self.request_timeout = self.config.get('request_timeout', 30)  # Request timeout in seconds

        # Search parameters
        self.keywords = self.config.get('keywords', ['强化学习', 'Code Agent', 'Agentic', '大模型'])
        self.search_type = '_2'  # Search articles only
        self.sort_type = '_2'  # Sort by latest

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; research-bot/1.0)'
        })

    def fetch_articles(self) -> List[Dict[str, Any]]:
        """Fetch articles from WeChat using WeChat Search API with time filtering and deduplication"""
        all_articles = []
        seen_doc_ids = set()  # Track seen docIDs for deduplication
        seen_urls = set()  # Track seen URLs for additional deduplication

        if not self.api_token:
            print("Warning: No API token provided for WeChat search")
            return all_articles

        # Calculate date range for the last 1 days
        end_date = datetime.now()
        start_date = datetime.fromtimestamp(end_date.timestamp() - 1 * 24 * 3600)

        print(
            f"Fetching WeChat articles from {start_date.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')}")

        for keyword in self.keywords:
            try:
                # Search and filter articles in one step
                filtered_articles = self._search_wechat_articles(keyword, start_date, end_date)
                # Deduplicate articles
                deduplicated_articles = self._deduplicate_articles(filtered_articles, seen_doc_ids, seen_urls)
                all_articles.extend(deduplicated_articles)
                print(f"Fetched {len(deduplicated_articles)} unique articles for keyword '{keyword}'")
            except Exception as e:
                print(f"Error searching WeChat with keyword '{keyword}': {e}")
                continue

        # Sort articles by timestamp (newest first)
        all_articles.sort(key=lambda x: x.get('timestamp', 0), reverse=True)

        print(f"Total unique articles fetched: {len(all_articles)}")

        # Add summary statistics
        if all_articles:
            timestamps = [article.get('timestamp', 0) for article in all_articles if article.get('timestamp', 0)]
            if timestamps:
                newest = datetime.fromtimestamp(max(timestamps)).strftime('%Y-%m-%d %H:%M:%S')
                oldest = datetime.fromtimestamp(min(timestamps)).strftime('%Y-%m-%d %H:%M:%S')
                print(f"Article date range: {oldest} to {newest}")

        return all_articles

    def _search_wechat_articles(self, keyword: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Search WeChat articles using WeChat Search API with offset pagination, retry logic, and in-fetch filtering"""
        filtered_articles = []
        offset = 0
        limit = 20  # API returns 20 items per page
        max_retries = self.max_retries
        base_delay = self.retry_base_delay  # Base delay for exponential backoff (seconds)

        # Convert datetime objects to Unix timestamps for comparison
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())

        print(f"Searching articles for keyword '{keyword}' with timestamp range: {start_timestamp} to {end_timestamp}")

        total_fetched = 0
        total_filtered = 0
        consecutive_low_filter = 0  # Track consecutive pages with low filtered results

        while True:
            # Build API URL with new parameters
            params = {
                'token': self.api_token,
                'keyword': keyword,
                'offset': offset,
                'searchType': self.search_type,
                'sortType': self.sort_type
            }

            api_url = f"{self.api_base_url}{self.api_endpoint}"

            # Retry mechanism with exponential backoff
            retry_count = 0
            success = False
            page_articles = []  # Store articles from this page
            page_filtered_articles = []  # Store filtered articles from this page

            while retry_count < max_retries and not success:
                try:
                    response = self.session.get(api_url, params=params, timeout=self.request_timeout)
                    response.raise_for_status()

                    data = response.json()

                    # Check API response code
                    if data.get('code') != 0:
                        error_msg = data.get('message', 'Unknown error')
                        print(f"API error on attempt {retry_count + 1}: {error_msg}")

                        # Don't retry on certain API errors (like invalid token)
                        if 'token' in error_msg.lower() or 'invalid' in error_msg.lower():
                            print(f"Critical API error, stopping retries")
                            break

                        retry_count += 1
                        if retry_count < max_retries:
                            delay = base_delay * (2 ** retry_count)  # Exponential backoff
                            print(f"Retrying in {delay} seconds... (attempt {retry_count + 1}/{max_retries})")
                            time.sleep(delay)
                        continue

                    # Success - process the response
                    response_data = data.get('data', {})

                    # Extract articles from current page
                    page_articles = self._parse_api_response(response_data)

                    # Filter articles immediately by timestamp
                    for article in page_articles:
                        article_timestamp = article.get('timestamp', 0)
                        if article_timestamp and start_timestamp <= article_timestamp <= end_timestamp:
                            page_filtered_articles.append(article)
                        elif not article_timestamp:
                            # If no timestamp, include to be safe
                            page_filtered_articles.append(article)

                    success = True

                except requests.exceptions.Timeout as e:
                    retry_count += 1
                    print(f"Request timeout on attempt {retry_count}/{max_retries}: {e}")
                    if retry_count < max_retries:
                        delay = base_delay * (2 ** retry_count)
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)

                except requests.exceptions.ConnectionError as e:
                    retry_count += 1
                    print(f"Connection error on attempt {retry_count}/{max_retries}: {e}")
                    if retry_count < max_retries:
                        delay = base_delay * (2 ** retry_count)
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)

                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    print(f"API request failed on attempt {retry_count}/{max_retries}: {e}")
                    if retry_count < max_retries:
                        delay = base_delay * (2 ** retry_count)
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)

                except json.JSONDecodeError as e:
                    retry_count += 1
                    print(f"Failed to parse API response on attempt {retry_count}/{max_retries}: {e}")
                    if retry_count < max_retries:
                        delay = base_delay * (2 ** retry_count)
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)

                except Exception as e:
                    retry_count += 1
                    print(f"Unexpected error on attempt {retry_count}/{max_retries}: {e}")
                    if retry_count < max_retries:
                        delay = base_delay * (2 ** retry_count)
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)

            # If all retries failed, break the pagination loop
            if not success:
                print(f"Failed to fetch page after {max_retries} attempts, stopping pagination")
                break

            # Add filtered articles to results
            filtered_articles.extend(page_filtered_articles)
            total_fetched += len(page_articles)
            total_filtered += len(page_filtered_articles)

            # Check if we should continue pagination
            total_count = response_data.get('totalCount', 0) if success else 0

            print(
                f"Fetched {len(page_articles)} articles, filtered {len(page_filtered_articles)} (total fetched: {total_fetched}, filtered: {total_filtered}) for keyword '{keyword}'")

            # Track consecutive pages with low filtered results
            if len(page_filtered_articles) < limit * 0.3:  # Less than 30% of page limit
                consecutive_low_filter += 1
                print(f"Low filter rate detected: {consecutive_low_filter} consecutive pages")
            else:
                consecutive_low_filter = 0  # Reset counter if we get good results

            # Early termination logic: Since API sorts by latest (newest first),
            # if we get very few filtered articles from current page, subsequent pages
            # will have even fewer articles in our date range, so we can stop early

            # Key insight: if filtered articles < some threshold, we're moving out of date range
            # since API returns newest first, subsequent pages will be older and have even fewer matches
            if len(page_filtered_articles) < 3:  # Less than 3 articles in date range from this page
                print(f"Very few articles ({len(page_filtered_articles)}) in date range from this page, stopping early")
                break

            # Also check if the oldest article in this page is already outside our date range
            if len(page_articles) > 0:
                page_timestamps = [article.get('timestamp', 0) for article in page_articles if
                                   article.get('timestamp', 0)]
                if page_timestamps:
                    oldest_in_page = min(page_timestamps)
                    # If the oldest article in this page is older than our start date,
                    # subsequent pages will only have older articles, so we can stop
                    if oldest_in_page < start_timestamp:
                        print(f"Oldest article in page is older than start date, stopping pagination early")
                        break

            # Stop pagination conditions:
            # 1. Raw articles less than limit means we've reached the end of available data
            # 2. We've fetched the total count reported by API  
            # 3. We've reached our reasonable maximum limit
            if len(page_articles) < limit:
                print(f"Reached end of available data (got {len(page_articles)} < {limit} raw articles)")
                break
            elif total_fetched >= total_count:
                print(f"Fetched all {total_count} articles reported by API")
                break
            elif total_fetched >= 100:
                print(f"Reached maximum limit of 100 articles")
                break

            offset += limit

        print(
            f"Completed search for keyword '{keyword}': {total_filtered} articles in date range from {total_fetched} total articles")
        return filtered_articles

    def _parse_api_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse WeChat Search API response and convert to arXiv format"""
        articles = []

        # Parse the new API response structure with 'items' field
        items = data.get('items', []) if isinstance(data, dict) else []

        for item in items:
            article = self._convert_api_item_to_article(item)
            if article:
                articles.append(article)

        return articles

    def _convert_api_item_to_article(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert WeChat Search API item to arXiv format"""
        title = item.get('title', '')
        content = item.get('desc', '')  # Changed from 'content' to 'desc'
        link = item.get('doc_url', '')  # Changed from 'url' to 'doc_url'
        doc_id = item.get('docID', '')  # Original WeChat docID for deduplication

        # Handle date (Unix timestamp in seconds)
        pub_timestamp = item.get('timestamp', item.get('date', 0))
        pub_date = datetime.fromtimestamp(pub_timestamp).strftime('%Y-%m-%d %H:%M:%S') if pub_timestamp else ''

        # Handle source (author) - it's now a string in dateTime field
        source_info = item.get('source', {})
        author = source_info.get('title', 'WeChat') if isinstance(source_info, dict) else 'WeChat'

        if not title or not content:
            return None

        # Generate unique ID and clean content
        article_id = self._generate_wechat_id(title, pub_date)
        summary = self._clean_content(content)
        categories = self._categorize_content(title + " " + summary)

        return {
            "id": article_id,
            "categories": categories,
            "pdf": link or "",
            "abs": link or "",
            "authors": [author.strip() if author else "WeChat Research"],
            "title": title.strip(),
            "comment": f"Source: WeChat, Published: {pub_date or 'Unknown'}",
            "summary": summary,
            "timestamp": pub_timestamp,  # Store timestamp for filtering and sorting
            "doc_id": doc_id  # Store original docID for deduplication
        }

    def _generate_wechat_id(self, title: str, pub_date: str) -> str:
        """Generate unique ID for WeChat articles"""
        content = f"{title}_{pub_date or datetime.now().isoformat()}"
        hash_obj = hashlib.md5(content.encode())
        date_str = datetime.now().strftime("%y%m")
        return f"wechat.{date_str}.{hash_obj.hexdigest()[:8]}"

    def _clean_content(self, content: str) -> str:
        """Clean and truncate content"""
        if not content:
            return "No summary available"

        # Remove HTML tags and normalize whitespace
        import re
        clean_text = re.sub(r'<[^>]+>', '', content)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # Limit to 500 characters for consistency
        if len(clean_text) > 500:
            clean_text = clean_text[:497] + "..."

        return clean_text

    def _categorize_content(self, content: str) -> List[str]:
        """Categorize content based on Chinese keywords"""
        categories = ["wechat.article"]

        # Chinese keyword-based categorization for WeChat content
        ai_keywords = ['人工智能', '机器学习', '深度学习', '大模型', 'AI', '大语言模型', 'transformer', '神经网络']
        rl_keywords = ['强化学习', 'RL', 'Reinforcement Learning']
        cl_keywords = ['自然语言', '文本', '翻译', 'BERT', 'GPT', '语言模型', 'NLP', '分词']
        agent_keywords = ["Agent", "智能体"]

        content_lower = content.lower()

        if any(keyword in content_lower for keyword in ai_keywords):
            categories.append("wechat.ai")
        if any(keyword in content_lower for keyword in rl_keywords):
            categories.append("wechat.rl")
        if any(keyword in content_lower for keyword in cl_keywords):
            categories.append("wechat.cl")
        if any(keyword in content_lower for keyword in agent_keywords):
            categories.append("wechat.agent")

        return categories

    def _deduplicate_articles(self, articles: List[Dict[str, Any]], seen_doc_ids: set, seen_urls: set) -> List[
        Dict[str, Any]]:
        """Deduplicate articles based on docID and URL"""
        unique_articles = []

        for article in articles:
            # Get original docID from the article (stored during conversion)
            doc_id = article.get('doc_id', '')

            # Extract URL from the article
            url = article.get('pdf', '') or article.get('abs', '')
            url_hash = hashlib.md5(url.encode()).hexdigest() if url else None

            # Skip if we've seen this docID before (primary deduplication)
            if doc_id and doc_id in seen_doc_ids:
                print(f"Skipping duplicate article by docID: {article.get('title', 'Unknown')[:50]}...")
                continue

            # Skip if we've seen this URL before (secondary deduplication)
            if url_hash and url_hash in seen_urls:
                print(f"Skipping duplicate article by URL: {article.get('title', 'Unknown')[:50]}...")
                continue

            # Add to seen sets
            if doc_id:
                seen_doc_ids.add(doc_id)
            if url_hash:
                seen_urls.add(url_hash)

            # Remove internal fields from final output as they're not needed in the arXiv format
            clean_article = {k: v for k, v in article.items() if k not in ['doc_id', 'timestamp']}
            unique_articles.append(clean_article)

        return unique_articles

    def _filter_articles_by_timestamp(self, articles: List[Dict[str, Any]], start_date: datetime, end_date: datetime) -> \
            List[Dict[str, Any]]:
        """Filter articles by timestamp range"""
        filtered_articles = []

        # Convert datetime objects to Unix timestamps for comparison
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())

        print(f"Filtering articles by timestamp range: {start_timestamp} to {end_timestamp}")

        for article in articles:
            # Get timestamp from article (direct from API response)
            article_timestamp = article.get('timestamp', 0)

            if article_timestamp:
                # Check if article timestamp is within the specified range
                if start_timestamp <= article_timestamp <= end_timestamp:
                    filtered_articles.append(article)
            else:
                # If no timestamp found, include the article to be safe
                print(
                    f"Warning: Article '{article.get('title', 'Unknown')[:50]}...' has no timestamp, including anyway")
                filtered_articles.append(article)

        print(f"Filtered {len(filtered_articles)} articles from {len(articles)} total articles")
        return filtered_articles


def main():
    current_file_path = os.path.abspath(__file__)
    print(f"当前文件路径: {current_file_path}")

    # 获取当前文件所在目录
    current_dir = os.path.dirname(current_file_path)
    print(f"当前文件所在目录: {current_dir}")

    # 获取上一级目录
    parent_dir = os.path.dirname(current_dir)
    print(f"上一级目录: {parent_dir}")

    # 获取上两级目录
    grandgrandparent_dir = os.path.dirname(os.path.dirname(parent_dir))

    print(f'dir:{grandgrandparent_dir}')
    """Main function to fetch WeChat articles"""
    fetcher = WeChatFetcher()
    articles = fetcher.fetch_articles()

    # Output as JSONL
    today = datetime.now().strftime("%Y-%m-%d")
    # start_date = datetime.fromtimestamp(datetime.now().timestamp() - 1 * 24 * 3600).strftime("%Y-%m-%d")
    # 获取当前文件的绝对路径

    output_file = os.path.join(grandgrandparent_dir, "data/{today}_wechat.jsonl")

    os.makedirs("data", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

    print(f"Fetched {len(articles)} WeChat articles to {output_file}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
WeChat Data Fetcher
Fetches articles from WeChat sources using Unified Search API and converts them to arXiv format
"""

import json
import os
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Any
import requests


class WeChatFetcher:
    """Fetches and processes WeChat articles using Unified Search API to match arXiv format"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # WeChat Search API configuration - updated endpoint
        self.api_base_url = self.config.get('api_base_url', 'http://47.117.133.51:30015')
        self.api_endpoint = '/api/weixin/search/v1'  # New endpoint
        self.api_token = self.config.get('api_token', os.environ.get('WECHAT_API_TOKEN', ''))

        # Retry configuration
        self.max_retries = self.config.get('max_retries', 5)
        self.retry_base_delay = self.config.get('retry_base_delay', 1)  # Base delay in seconds
        self.request_timeout = self.config.get('request_timeout', 30)  # Request timeout in seconds

        # Search parameters
        self.keywords = self.config.get('keywords', ['强化学习', 'Code Agent', 'Agentic', '大模型'])
        self.search_type = '_2'  # Search articles only
        self.sort_type = '_2'  # Sort by latest

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; research-bot/1.0)'
        })

    def fetch_articles(self) -> List[Dict[str, Any]]:
        """Fetch articles from WeChat using WeChat Search API with time filtering and deduplication"""
        all_articles = []
        seen_doc_ids = set()  # Track seen docIDs for deduplication
        seen_urls = set()  # Track seen URLs for additional deduplication

        if not self.api_token:
            print("Warning: No API token provided for WeChat search")
            return all_articles

        # Calculate date range for the last 1 days
        end_date = datetime.now()
        start_date = datetime.fromtimestamp(end_date.timestamp() - 1 * 24 * 3600)

        print(
            f"Fetching WeChat articles from {start_date.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')}")

        for keyword in self.keywords:
            try:
                # Search and filter articles in one step
                filtered_articles = self._search_wechat_articles(keyword, start_date, end_date)
                # Deduplicate articles
                deduplicated_articles = self._deduplicate_articles(filtered_articles, seen_doc_ids, seen_urls)
                all_articles.extend(deduplicated_articles)
                print(f"Fetched {len(deduplicated_articles)} unique articles for keyword '{keyword}'")
            except Exception as e:
                print(f"Error searching WeChat with keyword '{keyword}': {e}")
                continue

        # Sort articles by timestamp (newest first)
        all_articles.sort(key=lambda x: x.get('timestamp', 0), reverse=True)

        print(f"Total unique articles fetched: {len(all_articles)}")

        # Add summary statistics
        if all_articles:
            timestamps = [article.get('timestamp', 0) for article in all_articles if article.get('timestamp', 0)]
            if timestamps:
                newest = datetime.fromtimestamp(max(timestamps)).strftime('%Y-%m-%d %H:%M:%S')
                oldest = datetime.fromtimestamp(min(timestamps)).strftime('%Y-%m-%d %H:%M:%S')
                print(f"Article date range: {oldest} to {newest}")

        return all_articles

    def _search_wechat_articles(self, keyword: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Search WeChat articles using WeChat Search API with offset pagination, retry logic, and in-fetch filtering"""
        filtered_articles = []
        offset = 0
        limit = 20  # API returns 20 items per page
        max_retries = self.max_retries
        base_delay = self.retry_base_delay  # Base delay for exponential backoff (seconds)

        # Convert datetime objects to Unix timestamps for comparison
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())

        print(f"Searching articles for keyword '{keyword}' with timestamp range: {start_timestamp} to {end_timestamp}")

        total_fetched = 0
        total_filtered = 0
        consecutive_low_filter = 0  # Track consecutive pages with low filtered results

        while True:
            # Build API URL with new parameters
            params = {
                'token': self.api_token,
                'keyword': keyword,
                'offset': offset,
                'searchType': self.search_type,
                'sortType': self.sort_type
            }

            api_url = f"{self.api_base_url}{self.api_endpoint}"

            # Retry mechanism with exponential backoff
            retry_count = 0
            success = False
            page_articles = []  # Store articles from this page
            page_filtered_articles = []  # Store filtered articles from this page

            while retry_count < max_retries and not success:
                try:
                    response = self.session.get(api_url, params=params, timeout=self.request_timeout)
                    response.raise_for_status()

                    data = response.json()

                    # Check API response code
                    if data.get('code') != 0:
                        error_msg = data.get('message', 'Unknown error')
                        print(f"API error on attempt {retry_count + 1}: {error_msg}")

                        # Don't retry on certain API errors (like invalid token)
                        if 'token' in error_msg.lower() or 'invalid' in error_msg.lower():
                            print(f"Critical API error, stopping retries")
                            break

                        retry_count += 1
                        if retry_count < max_retries:
                            delay = base_delay * (2 ** retry_count)  # Exponential backoff
                            print(f"Retrying in {delay} seconds... (attempt {retry_count + 1}/{max_retries})")
                            time.sleep(delay)
                        continue

                    # Success - process the response
                    response_data = data.get('data', {})

                    # Extract articles from current page
                    page_articles = self._parse_api_response(response_data)

                    # Filter articles immediately by timestamp
                    for article in page_articles:
                        article_timestamp = article.get('timestamp', 0)
                        if article_timestamp and start_timestamp <= article_timestamp <= end_timestamp:
                            page_filtered_articles.append(article)
                        elif not article_timestamp:
                            # If no timestamp, include to be safe
                            page_filtered_articles.append(article)

                    success = True

                except requests.exceptions.Timeout as e:
                    retry_count += 1
                    print(f"Request timeout on attempt {retry_count}/{max_retries}: {e}")
                    if retry_count < max_retries:
                        delay = base_delay * (2 ** retry_count)
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)

                except requests.exceptions.ConnectionError as e:
                    retry_count += 1
                    print(f"Connection error on attempt {retry_count}/{max_retries}: {e}")
                    if retry_count < max_retries:
                        delay = base_delay * (2 ** retry_count)
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)

                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    print(f"API request failed on attempt {retry_count}/{max_retries}: {e}")
                    if retry_count < max_retries:
                        delay = base_delay * (2 ** retry_count)
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)

                except json.JSONDecodeError as e:
                    retry_count += 1
                    print(f"Failed to parse API response on attempt {retry_count}/{max_retries}: {e}")
                    if retry_count < max_retries:
                        delay = base_delay * (2 ** retry_count)
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)

                except Exception as e:
                    retry_count += 1
                    print(f"Unexpected error on attempt {retry_count}/{max_retries}: {e}")
                    if retry_count < max_retries:
                        delay = base_delay * (2 ** retry_count)
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)

            # If all retries failed, break the pagination loop
            if not success:
                print(f"Failed to fetch page after {max_retries} attempts, stopping pagination")
                break

            # Add filtered articles to results
            filtered_articles.extend(page_filtered_articles)
            total_fetched += len(page_articles)
            total_filtered += len(page_filtered_articles)

            # Check if we should continue pagination
            total_count = response_data.get('totalCount', 0) if success else 0

            print(
                f"Fetched {len(page_articles)} articles, filtered {len(page_filtered_articles)} (total fetched: {total_fetched}, filtered: {total_filtered}) for keyword '{keyword}'")

            # Track consecutive pages with low filtered results
            if len(page_filtered_articles) < limit * 0.3:  # Less than 30% of page limit
                consecutive_low_filter += 1
                print(f"Low filter rate detected: {consecutive_low_filter} consecutive pages")
            else:
                consecutive_low_filter = 0  # Reset counter if we get good results

            # Early termination logic: Since API sorts by latest (newest first),
            # if we get very few filtered articles from current page, subsequent pages
            # will have even fewer articles in our date range, so we can stop early

            # Key insight: if filtered articles < some threshold, we're moving out of date range
            # since API returns newest first, subsequent pages will be older and have even fewer matches
            if len(page_filtered_articles) < 3:  # Less than 3 articles in date range from this page
                print(f"Very few articles ({len(page_filtered_articles)}) in date range from this page, stopping early")
                break

            # Also check if the oldest article in this page is already outside our date range
            if len(page_articles) > 0:
                page_timestamps = [article.get('timestamp', 0) for article in page_articles if
                                   article.get('timestamp', 0)]
                if page_timestamps:
                    oldest_in_page = min(page_timestamps)
                    # If the oldest article in this page is older than our start date,
                    # subsequent pages will only have older articles, so we can stop
                    if oldest_in_page < start_timestamp:
                        print(f"Oldest article in page is older than start date, stopping pagination early")
                        break

            # Stop pagination conditions:
            # 1. Raw articles less than limit means we've reached the end of available data
            # 2. We've fetched the total count reported by API  
            # 3. We've reached our reasonable maximum limit
            if len(page_articles) < limit:
                print(f"Reached end of available data (got {len(page_articles)} < {limit} raw articles)")
                break
            elif total_fetched >= total_count:
                print(f"Fetched all {total_count} articles reported by API")
                break
            elif total_fetched >= 100:
                print(f"Reached maximum limit of 100 articles")
                break

            offset += limit

        print(
            f"Completed search for keyword '{keyword}': {total_filtered} articles in date range from {total_fetched} total articles")
        return filtered_articles

    def _parse_api_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse WeChat Search API response and convert to arXiv format"""
        articles = []

        # Parse the new API response structure with 'items' field
        items = data.get('items', []) if isinstance(data, dict) else []

        for item in items:
            article = self._convert_api_item_to_article(item)
            if article:
                articles.append(article)

        return articles

    def _convert_api_item_to_article(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert WeChat Search API item to arXiv format"""
        title = item.get('title', '')
        content = item.get('desc', '')  # Changed from 'content' to 'desc'
        link = item.get('doc_url', '')  # Changed from 'url' to 'doc_url'
        doc_id = item.get('docID', '')  # Original WeChat docID for deduplication

        # Handle date (Unix timestamp in seconds)
        pub_timestamp = item.get('timestamp', item.get('date', 0))
        pub_date = datetime.fromtimestamp(pub_timestamp).strftime('%Y-%m-%d %H:%M:%S') if pub_timestamp else ''

        # Handle source (author) - it's now a string in dateTime field
        source_info = item.get('source', {})
        author = source_info.get('title', 'WeChat') if isinstance(source_info, dict) else 'WeChat'

        if not title or not content:
            return None

        # Generate unique ID and clean content
        article_id = self._generate_wechat_id(title, pub_date)
        summary = self._clean_content(content)
        categories = self._categorize_content(title + " " + summary)

        return {
            "id": article_id,
            "categories": categories,
            "pdf": link or "",
            "abs": link or "",
            "authors": [author.strip() if author else "WeChat Research"],
            "title": title.strip(),
            "comment": f"Source: WeChat, Published: {pub_date or 'Unknown'}",
            "summary": summary,
            "timestamp": pub_timestamp,  # Store timestamp for filtering and sorting
            "doc_id": doc_id  # Store original docID for deduplication
        }

    def _generate_wechat_id(self, title: str, pub_date: str) -> str:
        """Generate unique ID for WeChat articles"""
        content = f"{title}_{pub_date or datetime.now().isoformat()}"
        hash_obj = hashlib.md5(content.encode())
        date_str = datetime.now().strftime("%y%m")
        return f"wechat.{date_str}.{hash_obj.hexdigest()[:8]}"

    def _clean_content(self, content: str) -> str:
        """Clean and truncate content"""
        if not content:
            return "No summary available"

        # Remove HTML tags and normalize whitespace
        import re
        clean_text = re.sub(r'<[^>]+>', '', content)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # Limit to 500 characters for consistency
        if len(clean_text) > 500:
            clean_text = clean_text[:497] + "..."

        return clean_text

    def _categorize_content(self, content: str) -> List[str]:
        """Categorize content based on Chinese keywords"""
        categories = ["wechat.article"]

        # Chinese keyword-based categorization for WeChat content
        ai_keywords = ['人工智能', '机器学习', '深度学习', '大模型', 'AI', '大语言模型', 'transformer', '神经网络']
        rl_keywords = ['强化学习', 'RL', 'Reinforcement Learning']
        cl_keywords = ['自然语言', '文本', '翻译', 'BERT', 'GPT', '语言模型', 'NLP', '分词']
        agent_keywords = ["Agent", "智能体"]

        content_lower = content.lower()

        if any(keyword in content_lower for keyword in ai_keywords):
            categories.append("wechat.ai")
        if any(keyword in content_lower for keyword in rl_keywords):
            categories.append("wechat.rl")
        if any(keyword in content_lower for keyword in cl_keywords):
            categories.append("wechat.cl")
        if any(keyword in content_lower for keyword in agent_keywords):
            categories.append("wechat.agent")

        return categories

    def _deduplicate_articles(self, articles: List[Dict[str, Any]], seen_doc_ids: set, seen_urls: set) -> List[
        Dict[str, Any]]:
        """Deduplicate articles based on docID and URL"""
        unique_articles = []

        for article in articles:
            # Get original docID from the article (stored during conversion)
            doc_id = article.get('doc_id', '')

            # Extract URL from the article
            url = article.get('pdf', '') or article.get('abs', '')
            url_hash = hashlib.md5(url.encode()).hexdigest() if url else None

            # Skip if we've seen this docID before (primary deduplication)
            if doc_id and doc_id in seen_doc_ids:
                print(f"Skipping duplicate article by docID: {article.get('title', 'Unknown')[:50]}...")
                continue

            # Skip if we've seen this URL before (secondary deduplication)
            if url_hash and url_hash in seen_urls:
                print(f"Skipping duplicate article by URL: {article.get('title', 'Unknown')[:50]}...")
                continue

            # Add to seen sets
            if doc_id:
                seen_doc_ids.add(doc_id)
            if url_hash:
                seen_urls.add(url_hash)

            # Remove internal fields from final output as they're not needed in the arXiv format
            clean_article = {k: v for k, v in article.items() if k not in ['doc_id', 'timestamp']}
            unique_articles.append(clean_article)

        return unique_articles

    def _filter_articles_by_timestamp(self, articles: List[Dict[str, Any]], start_date: datetime, end_date: datetime) -> \
            List[Dict[str, Any]]:
        """Filter articles by timestamp range"""
        filtered_articles = []

        # Convert datetime objects to Unix timestamps for comparison
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())

        print(f"Filtering articles by timestamp range: {start_timestamp} to {end_timestamp}")

        for article in articles:
            # Get timestamp from article (direct from API response)
            article_timestamp = article.get('timestamp', 0)

            if article_timestamp:
                # Check if article timestamp is within the specified range
                if start_timestamp <= article_timestamp <= end_timestamp:
                    filtered_articles.append(article)
            else:
                # If no timestamp found, include the article to be safe
                print(
                    f"Warning: Article '{article.get('title', 'Unknown')[:50]}...' has no timestamp, including anyway")
                filtered_articles.append(article)

        print(f"Filtered {len(filtered_articles)} articles from {len(articles)} total articles")
        return filtered_articles


def main():
    current_file_path = os.path.abspath(__file__)
    print(f"当前文件路径: {current_file_path}")

    # 获取当前文件所在目录
    current_dir = os.path.dirname(current_file_path)
    print(f"当前文件所在目录: {current_dir}")

    # 获取上一级目录
    parent_dir = os.path.dirname(current_dir)
    print(f"上一级目录: {parent_dir}")

    # 获取上两级目录
    grandgrandparent_dir = os.path.dirname(os.path.dirname(parent_dir))

    print(f'dir:{grandgrandparent_dir}')
    """Main function to fetch WeChat articles"""
    fetcher = WeChatFetcher()
    articles = fetcher.fetch_articles()

    # Output as JSONL
    today = datetime.now().strftime("%Y-%m-%d")
    # start_date = datetime.fromtimestamp(datetime.now().timestamp() - 1 * 24 * 3600).strftime("%Y-%m-%d")
    # 获取当前文件的绝对路径

    output_file = os.path.join(grandgrandparent_dir, f"data/{today}_wechat.jsonl")

    os.makedirs("data", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

    print(f"Fetched {len(articles)} WeChat articles to {output_file}")


if __name__ == "__main__":
    main()
