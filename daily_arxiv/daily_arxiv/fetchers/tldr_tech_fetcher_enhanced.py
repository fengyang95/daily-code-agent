#!/usr/bin/env python3
"""
TLDR Tech Newsletter Fetcher - Core Version
Extracts articles from TLDR newsletter emails with minimal dependencies
"""

import json
import os
import hashlib
import email
from datetime import datetime
from typing import List, Dict, Any
from email.utils import parsedate_to_datetime
import re

try:
    from .email_client import Email163Client
except ImportError:
    from email_client import Email163Client


class TLDRTechFetcher:
    """Core TLDR tech newsletter fetcher - extracts articles from email content"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Email configuration
        self.email_address = self.config.get('email_address',
                                             os.environ.get('EMAIL_163_ADDRESS', 'xxx'))
        self.email_password = self.config.get('email_password',
                                              os.environ.get('EMAIL_163_PASSWORD', 'xxx'))

        # Sender filtering
        sender_filter = self.config.get('sender_filter', os.environ.get('TLDR_SENDER_FILTER', 'xxx'))
        if isinstance(sender_filter, str):
            self.target_senders = sender_filter.strip().split(',')
        elif isinstance(sender_filter, list):
            self.target_senders = [s.strip() for s in sender_filter if s.strip()]
        else:
            self.target_senders = ['xxx']

        # Search configuration
        self.search_days = self.config.get('search_days', 7)
        self.max_articles = self.config.get('max_articles', 100)

        # Initialize email client
        self.email_client = None
        if self.email_address and self.email_password:
            self.email_client = Email163Client(self.email_address, self.email_password)

    def fetch_articles(self) -> List[Dict[str, Any]]:
        """Fetch TLDR articles from email"""
        if not self.email_client:
            print("No email credentials provided")
            return []

        articles = []

        # Try IMAP first
        try:
            print("Connecting via IMAP...")
            articles = self._fetch_via_imap()
            if articles:
                return articles[:self.max_articles]
        except Exception as e:
            print(f"IMAP failed: {e}")

        # Fallback to POP3
        try:
            print("Trying POP3 fallback...")
            articles = self._fetch_via_pop3()
            if articles:
                return articles[:self.max_articles]
        except Exception as e:
            print(f"POP3 also failed: {e}")

        return articles[:self.max_articles]

    def _fetch_via_imap(self) -> List[Dict[str, Any]]:
        """Fetch articles using IMAP"""
        articles = []
        mail = None

        try:
            mail = self.email_client.connect_imap()
            if not self.email_client.select_inbox(mail):
                raise Exception("Failed to select inbox")

            # Search for TLDR emails
            email_ids = self.email_client.search_emails(mail, self.target_senders, self.search_days)
            print(f"Found {len(email_ids)} TLDR emails")

            # Process each email
            for email_id in email_ids:
                try:
                    raw_email = self.email_client.fetch_email_content(mail, email_id)
                    if raw_email:
                        email_message = email.message_from_bytes(raw_email)
                        email_articles = self._extract_articles_from_email(email_message)
                        articles.extend(email_articles)
                except Exception as e:
                    print(f"Error processing email: {e}")
                    continue

        finally:
            if mail:
                self.email_client.close_imap(mail)

        return articles

    def _fetch_via_pop3(self) -> List[Dict[str, Any]]:
        """Fetch articles using POP3"""
        articles = []
        mail = None

        try:
            mail = self.email_client.connect_pop3()
            msg_count = self.email_client.get_pop3_message_count(mail)
            print(f"Found {msg_count} total messages")

            # Process recent messages (last 50 for performance)
            start_msg = max(1, msg_count - 49)

            for msg_num in range(start_msg, msg_count + 1):
                try:
                    raw_email = self.email_client.fetch_pop3_email(mail, msg_num)
                    if raw_email:
                        email_message = email.message_from_bytes(raw_email)
                        email_articles = self._extract_articles_from_email(email_message)
                        articles.extend(email_articles)
                except Exception as e:
                    print(f"Error processing message {msg_num}: {e}")
                    continue

        finally:
            if mail:
                self.email_client.close_pop3(mail)

        return articles

    def _extract_articles_from_email(self, email_message) -> List[Dict[str, Any]]:
        """Extract TLDR articles from email message"""
        # Check sender
        from_sender = self._decode_header(email_message.get('From', ''))
        if not self._is_sender_match(from_sender):
            return []

        # Get email date
        date_str = email_message.get('Date', '')
        try:
            email_date = parsedate_to_datetime(date_str)
        except:
            email_date = datetime.now()

        # Extract body content
        body = self._extract_email_body(email_message)
        if not body:
            return []

        # Parse TLDR content
        articles = self._parse_tldr_content(body, email_date)
        return articles

    def _parse_tldr_content(self, content: str, email_date: datetime) -> List[Dict[str, Any]]:
        """Parse TLDR content using asterisk-based title format"""
        articles = []

        # Check if content is HTML - improved detection
        is_html = ('<html>' in content.lower() or '<body>' in content.lower() or
                   '<table' in content.lower() or '<div' in content.lower() or
                   '<a href=' in content.lower() or '<strong>' in content.lower())

        if is_html:
            # For HTML content, extract links from <a> tags and convert to plain text format
            content = self._extract_links_from_html(content)

        # Parse using asterisk title format: *Title* (info) <link>
        lines = content.split('\n')
        current_article = None
        current_summary_lines = []

        # Regex patterns - make title matching more flexible
        title_pattern = re.compile(r'\*([^*]+)\*')  # *title* (original format)
        title_pattern2 = re.compile(r'^([^*\n]+?)\s*<([^>]+)>')  # Title <link>
        title_pattern3 = re.compile(r'<strong>([^<]+)</strong>')  # <strong>Title</strong> (HTML format)
        title_pattern4 = re.compile(r'<strong[^>]*>([^<]+)</strong>')  # <strong>Title</strong> with attributes
        min_read_pattern = re.compile(r'\((\d+\s*minute?\s+read)\)')  # (X minute read)
        sponsor_pattern = re.compile(r'\(([^)]*[Ss]ponsor[^)]*)\)')  # (Sponsor)
        link_pattern = re.compile(r'<(https?://[^>\s]+)>')  # <URL> - only capture actual URLs
        paren_pattern = re.compile(r'\([^)]*\)')  # anything in parentheses
        url_pattern = re.compile(r'https?://[^\s<>]+')  # Direct URL pattern

        title_matches = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for title (*xxx*)
            title_match = title_pattern.search(line)
            title_match2 = title_pattern2.search(line)
            title_match3 = title_pattern3.search(line)
            title_match4 = title_pattern4.search(line)

            if title_match or title_match2 or title_match3 or title_match4:
                title_matches += 1
                if title_match:
                    title_text = title_match.group(1).strip()
                elif title_match2:
                    title_text = title_match2.group(1).strip()
                elif title_match3:
                    title_text = title_match3.group(1).strip()
                else:
                    title_text = title_match4.group(1).strip()

                # Filter out invalid titles (email headers, navigation, etc.)
                if self._is_invalid_title(title_text):
                    continue

                # Save previous article if exists
                if current_article and current_summary_lines:
                    current_article['summary'] = self._clean_summary(' '.join(current_summary_lines))
                    final_article = self._create_article(
                        current_article['title'],
                        current_article['summary'],
                        email_date,
                        current_article['reading_info'],
                        current_article.get('links', [])
                    )
                    if final_article:
                        articles.append(final_article)

                # Start new article
                # Extract reading info and links
                min_read_match = min_read_pattern.search(line)
                sponsor_match = sponsor_pattern.search(line)
                links = link_pattern.findall(line)
                direct_urls = url_pattern.findall(line)
                links.extend(direct_urls)  # Add direct URLs to links list

                reading_info = None
                if min_read_match:
                    reading_info = min_read_match.group(1)
                elif sponsor_match:
                    reading_info = sponsor_match.group(1)

                # Clean title - remove HTML tags and entities
                clean_title = self._clean_title(title_text, min_read_pattern, sponsor_pattern, paren_pattern)
                clean_title = re.sub(r'<[^>]+>', '', clean_title)  # Remove HTML tags
                clean_title = re.sub(r'&#39;', "'", clean_title)  # Fix HTML entities
                clean_title = re.sub(r'&quot;', '"', clean_title)
                clean_title = re.sub(r'&amp;', '&', clean_title)
                clean_title = clean_title.strip()

                current_article = {
                    'title': clean_title,
                    'reading_info': reading_info,
                    'links': links,
                }
                current_summary_lines = []

            elif current_article is not None:
                # This line is part of the summary
                links_in_line = link_pattern.findall(line)
                direct_urls_in_line = url_pattern.findall(line)
                if links_in_line or direct_urls_in_line:
                    current_article['links'].extend(links_in_line)
                    current_article['links'].extend(direct_urls_in_line)

                # Clean the line for summary - be more permissive
                cleaned_line = link_pattern.sub('', line).strip()
                if cleaned_line and len(cleaned_line) > 5:  # Basic length check
                    current_summary_lines.append(cleaned_line)

        # Don't forget the last article
        if current_article and current_summary_lines:
            current_article['summary'] = self._clean_summary(' '.join(current_summary_lines))
            final_article = self._create_article(
                current_article['title'],
                current_article['summary'],
                email_date,
                current_article['reading_info'],
                current_article.get('links', [])
            )
            if final_article:
                articles.append(final_article)

        return articles

    def _extract_links_from_html(self, html_content: str) -> str:
        """Extract links from HTML and convert to plain text format"""
        from bs4 import BeautifulSoup

        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Convert HTML to plain text, but preserve link information inline
            text_content = ""
            seen_titles = set()  # 避免重复

            # Find all strong tags that likely contain article titles
            for strong_tag in soup.find_all('strong'):
                title_text = strong_tag.get_text().strip()

                # Skip if title is too short or invalid
                if (len(title_text) < 10 or
                        title_text in seen_titles or
                        self._is_invalid_title(title_text)):
                    continue

                # Find the nearest parent link
                parent = strong_tag.parent
                link_url = None

                # Look for a link in the same container or parent
                if parent and parent.name == 'a' and parent.get('href'):
                    link_url = parent.get('href')
                else:
                    # Search for links in the same table cell or container
                    container = strong_tag.find_parent(['td', 'div', 'p'])
                    if container:
                        link_tag = container.find('a', href=True)
                        if link_tag:
                            link_url = link_tag.get('href')

                # Only include HTTP/HTTPS links
                if link_url and link_url.startswith(('http://', 'https://')):
                    seen_titles.add(title_text)
                    # Convert to TLDR format: *Title* <link>
                    text_content += f"*{title_text}* <{link_url}>\n"

                    # Find summary text from the same container
                    if container:
                        # Get all text from container, excluding other links
                        summary_parts = []
                        for element in container.contents:
                            if hasattr(element, 'name'):
                                if element.name == 'a':
                                    continue  # Skip links
                                elif element.name == 'strong':
                                    continue  # Skip the title we already processed
                                elif hasattr(element, 'get_text'):
                                    text = element.get_text().strip()
                                    if text and len(text) > 10:
                                        summary_parts.append(text)
                            elif isinstance(element, str) and element.strip():
                                text = element.strip()
                                if len(text) > 10:
                                    summary_parts.append(text)

                        if summary_parts:
                            text_content += ' '.join(summary_parts) + "\n"

                    text_content += "\n"

            # If we didn't find structured content, fall back to simple extraction
            if not text_content.strip():
                text_content = soup.get_text()

            return text_content.strip()

        except Exception as e:
            print(f"Error parsing HTML: {e}")
            # Fallback: just extract text
            return re.sub(r'<[^>]+>', '', html_content)

    def _create_article(self, title: str, summary: str, email_date: datetime,
                        reading_info: str = None, links: list = None) -> Dict[str, Any]:
        """Create TLDR article in arXiv format"""
        if not title or not summary:
            return None

        # Filter out valid HTTP/HTTPS links and clean them
        valid_links = []
        for link in (links or []):
            # Clean up the link - remove any trailing punctuation or whitespace
            clean_link = link.strip().rstrip('.,;:!?')
            if clean_link.startswith(('http://', 'https://')):
                valid_links.append(clean_link)

        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in valid_links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)

        # Generate unique ID
        article_id = self._generate_id(title, email_date)

        # Build comment
        comment_parts = [f"Source: TLDR Newsletter, Date: {email_date.strftime('%Y-%m-%d')}"]
        if reading_info:
            comment_parts.append(f"Reading time: {reading_info}")
        if unique_links:
            comment_parts.append(f"Links: {', '.join(unique_links[:3])}")

        comment = ", ".join(comment_parts)

        # Determine primary link - prefer non-email, non-social media links
        primary_link = ""
        for link in unique_links:
            # Skip email links and common social media/share links
            if (not link.startswith('mailto:') and
                    not any(domain in link.lower() for domain in
                            ['twitter.com', 'x.com', 'facebook.com', 'linkedin.com', 'reddit.com', 't.co'])):
                primary_link = link
                break

        # If no good primary link found, use the first one
        if not primary_link and unique_links:
            primary_link = unique_links[0]

        return {
            "id": article_id,
            "categories": ["tldr.article"],
            "pdf": primary_link,
            "abs": primary_link,
            "authors": ["TLDR Newsletter"],
            "title": title.strip(),
            "comment": comment,
            "summary": summary,
            "source": "tldr"
        }

    def _extract_email_body(self, email_message) -> str:
        """Extract text content from email, preferring HTML for better link extraction"""
        html_body = ""
        text_body = ""

        if email_message.is_multipart():
            for part in email_message.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                if "attachment" in content_disposition:
                    continue

                try:
                    part_body = part.get_payload(decode=True).decode('utf-8', errors='ignore')

                    if content_type == "text/html":
                        html_body += part_body + "\n"
                    elif content_type == "text/plain":
                        text_body += part_body + "\n"
                except:
                    continue
        else:
            try:
                content_type = email_message.get_content_type()
                body = email_message.get_payload(decode=True).decode('utf-8', errors='ignore')

                if content_type == "text/html":
                    html_body = body
                elif content_type == "text/plain":
                    text_body = body
            except:
                return ""

        # Prefer HTML body for better link extraction, fallback to text
        # Also check for forwarded content
        content = html_body if html_body else text_body

        # Try to extract forwarded content if this is a forwarded email
        if "---------- Forwarded message ---------" in content:
            # Find the start of forwarded content
            forwarded_start = content.find("---------- Forwarded message ---------")
            forwarded_content = content[forwarded_start:]

            # Remove the forwarded message header
            lines = forwarded_content.split('\n')
            content_lines = []
            header_ended = False

            for line in lines[1:]:  # Skip the "Forwarded message" line
                if not header_ended and (line.strip().startswith('From:') or
                                         line.strip().startswith('Date:') or
                                         line.strip().startswith('Subject:') or
                                         line.strip().startswith('To:')):
                    continue
                elif not header_ended and not line.strip():
                    header_ended = True
                    continue
                elif header_ended:
                    content_lines.append(line)

            if content_lines:
                content = '\n'.join(content_lines)

        return content

    def _decode_header(self, header: str) -> str:
        """Decode email header"""
        return header

    def _is_sender_match(self, sender: str) -> bool:
        """Check if sender matches target senders"""
        if not sender or not self.target_senders:
            return False

        sender_lower = sender.lower().strip()

        for target_sender in self.target_senders:
            target_lower = target_sender.lower().strip()
            if target_lower in sender_lower or sender_lower in target_lower:
                return True

        return False

    def _clean_title(self, title_text: str, min_read_pattern, sponsor_pattern, paren_pattern) -> str:
        """Clean TLDR title by removing parenthetical information"""
        cleaned = min_read_pattern.sub('', title_text)
        cleaned = sponsor_pattern.sub('', cleaned)
        cleaned = paren_pattern.sub('', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _clean_summary(self, summary: str) -> str:
        """Clean and truncate summary"""
        if not summary:
            return "No summary available"

        clean_text = re.sub(r'\s+', ' ', summary).strip()
        clean_text = re.sub(r'<[^>]*>', '', clean_text)

        # Fix HTML entities
        clean_text = re.sub(r'&#39;', "'", clean_text)
        clean_text = re.sub(r'&quot;', '"', clean_text)
        clean_text = re.sub(r'&amp;', '&', clean_text)
        clean_text = re.sub(r'&lt;', '<', clean_text)
        clean_text = re.sub(r'&gt;', '>', clean_text)

        if len(clean_text) > 500:
            clean_text = clean_text[:497] + "..."

        return clean_text

    def _is_invalid_title(self, title: str) -> bool:
        """Check if title is invalid (email headers, navigation, etc.)"""
        invalid_patterns = [
            r'^\s*-+',  # Lines starting with dashes
            r'^\s*forwarded\s+message',  # Forwarded message headers
            r'^\s*from:\s*',  # Email headers
            r'^\s*sign\s+up\s*$',  # Navigation (exact match)
            r'^\s*advertise\s*$',  # Navigation (exact match)
            r'^\s*view\s+online\s*$',  # Navigation (exact match)
            r'^\s*unsubscribe\s*$',  # Footer (exact match)
            r'^\s*manage\s+your\s+subscriptions\s*$',  # Footer (exact match)
            r'^\s*apply\s+here\s*$',  # Call to action (exact match)
            r'^[^@]*@',  # Email addresses (contains @)
            r'^\s*\d+\s*$',  # Just numbers
            r'^\s*together\s+with',  # Newsletter headers
            r'^\s*tldr\s+ai',  # Newsletter headers
            r'^\s*read\s+the\s+blog\s*$',  # Call to action
            r'^\s*join\s+live\s+on\s*$',  # Call to action
            r'^\s*ebook\s+by\s*$',  # Call to action
        ]

        title_lower = title.lower().strip()
        for pattern in invalid_patterns:
            if re.search(pattern, title_lower, re.IGNORECASE):
                return True

        # Too short or too long titles are likely invalid
        if len(title.strip()) < 3 or len(title.strip()) > 200:
            return True

        return False

    def _is_metadata_line(self, line: str) -> bool:
        """Check if line is metadata (section headers, etc.)"""
        line = line.strip()
        if not line:
            return True

        # Allow normal sentences that start with capital letters
        if re.match(r'^[A-Z][a-z\s]+$', line) and len(line) > 10:
            return False

        # Only filter out ALL CAPS lines
        if re.match(r'^[A-Z\s]+$', line) and len(line) > 2:
            return True
        if re.match(r'^[-=~*]+$', line.strip()):
            return True
        return False

    def _generate_id(self, title: str, date: datetime) -> str:
        """Generate unique ID for TLDR articles"""
        content = f"{title}_{date.isoformat()}"
        hash_obj = hashlib.md5(content.encode())
        date_str = date.strftime("%y%m")
        return f"tldr.{date_str}.{hash_obj.hexdigest()[:8]}"


def main():
    """Main function"""
    # Configuration
    config = {
        'search_days': 7,
        'max_articles': 500
    }

    fetcher = TLDRTechFetcher(config)
    articles = fetcher.fetch_articles()

    # Output as JSONL
    today = datetime.now().strftime("%Y-%m-%d")
    output_file = f"data/{today}_tldr.jsonl"

    os.makedirs("data", exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

    print(f"Fetched {len(articles)} TLDR articles to {output_file}")


if __name__ == "__main__":
    main()
