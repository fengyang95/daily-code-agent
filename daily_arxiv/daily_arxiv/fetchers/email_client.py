#!/usr/bin/env python3
"""
Email Client for 163 Email Service
Handles IMAP/POP3 connections and authentication
"""

import imaplib
import poplib
import ssl
import re
from typing import List, Optional

class Email163Client:
    """163 Email client for IMAP/POP3 operations"""
    
    def __init__(self, email_address: str, email_password: str):
        self.email_address = email_address
        self.email_password = email_password
        self.imap_server = 'imap.163.com'
        self.imap_port = 993
        self.pop_server = 'pop.163.com'
        self.pop_port = 995
    
    def connect_imap(self) -> Optional[imaplib.IMAP4_SSL]:
        """Connect to IMAP server with authentication"""
        try:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            mail = imaplib.IMAP4_SSL(self.imap_server, self.imap_port, ssl_context=context)
            
            # Try standard login first
            try:
                mail.login(self.email_address, self.email_password)
                return mail
            except Exception as e:
                # Try PLAIN authentication
                try:
                    mail.authenticate('PLAIN', lambda x: f'\x00{self.email_address}\x00{self.email_password}'.encode())
                    return mail
                except:
                    raise e
                    
        except Exception as e:
            raise Exception(f"IMAP connection failed: {str(e)}")
    
    def connect_pop3(self) -> Optional[poplib.POP3_SSL]:
        """Connect to POP3 server with authentication"""
        try:
            mail = poplib.POP3_SSL(self.pop_server, self.pop_port)
            mail.user(self.email_address)
            mail.pass_(self.email_password)
            return mail
        except Exception as e:
            error_msg = str(e)
            # Decode Chinese error messages if present
            if 'b\'' in error_msg:
                try:
                    import ast
                    byte_match = re.search(r"b'([^']*)'", error_msg)
                    if byte_match:
                        chinese_bytes = ast.literal_eval(f"b'{byte_match.group(1)}'")
                        decoded_msg = chinese_bytes.decode('gbk', errors='ignore')
                        raise Exception(f"POP3 authentication failed: {decoded_msg}")
                except:
                    pass
            raise Exception(f"POP3 connection failed: {str(e)}")
    
    def select_inbox(self, mail: imaplib.IMAP4_SSL) -> bool:
        """Select inbox folder with fallback options"""
        folders = ['INBOX', 'inbox']
        
        for folder in folders:
            try:
                status, response = mail.select(folder)
                if status == 'OK':
                    return True
            except:
                continue
        
        # Try to list and find inbox
        try:
            status, folder_list = mail.list()
            if status == 'OK':
                for folder_info in folder_list:
                    try:
                        folder_str = folder_info.decode('utf-8', errors='ignore')
                        parts = folder_str.split(' "/" ')
                        if len(parts) >= 2:
                            folder_name = parts[-1].strip('"')
                            if folder_name.upper() in ['INBOX', '收件箱'] or '收件' in folder_name:
                                status, response = mail.select(f'"{folder_name}"')
                                if status == 'OK':
                                    return True
                    except:
                        continue
        except:
            pass
        
        return False
    
    def search_emails(self, mail: imaplib.IMAP4_SSL, target_senders: List[str], 
                     since_days: int = 7) -> List[bytes]:
        """Search for emails from target senders"""
        since_date = self._get_since_date(since_days)
        email_ids = []
        
        # Try searching for each sender
        for sender in target_senders:
            try:
                search_criteria = f'(FROM "{sender}" SINCE "{since_date}")'
                status, message_numbers = mail.search(None, search_criteria)
                if status == 'OK' and message_numbers and message_numbers[0]:
                    found_ids = message_numbers[0].split()
                    email_ids.extend(found_ids)
            except:
                continue
        
        # If no results, try without date filter
        if not email_ids:
            for sender in target_senders:
                try:
                    search_criteria = f'(FROM "{sender}")'
                    status, message_numbers = mail.search(None, search_criteria)
                    if status == 'OK' and message_numbers and message_numbers[0]:
                        found_ids = message_numbers[0].split()
                        email_ids.extend(found_ids)
                except:
                    continue
        
        # Final fallback: get all emails and filter manually
        if not email_ids:
            try:
                status, message_numbers = mail.search(None, 'ALL')
                if status == 'OK' and message_numbers and message_numbers[0]:
                    all_ids = message_numbers[0].split()
                    email_ids = self._manual_filter_by_sender(mail, all_ids, target_senders, since_days)
            except:
                pass
        
        return list(set(email_ids))  # Remove duplicates
    
    def fetch_email_content(self, mail: imaplib.IMAP4_SSL, email_id: bytes) -> Optional[bytes]:
        """Fetch full email content"""
        try:
            status, msg_data = mail.fetch(email_id, '(RFC822)')
            if status == 'OK':
                return msg_data[0][1]
        except:
            pass
        return None
    
    def get_pop3_message_count(self, mail: poplib.POP3_SSL) -> int:
        """Get total number of messages in POP3 mailbox"""
        try:
            return len(mail.list()[1])
        except:
            return 0
    
    def fetch_pop3_email(self, mail: poplib.POP3_SSL, msg_num: int) -> Optional[bytes]:
        """Fetch single POP3 email"""
        try:
            response = mail.retr(msg_num)
            return b'\n'.join(response[1])
        except:
            return None
    
    def close_imap(self, mail: imaplib.IMAP4_SSL):
        """Close IMAP connection"""
        try:
            mail.close()
            mail.logout()
        except:
            pass
    
    def close_pop3(self, mail: poplib.POP3_SSL):
        """Close POP3 connection"""
        try:
            mail.quit()
        except:
            pass
    
    def _get_since_date(self, days: int) -> str:
        """Get date string for IMAP search"""
        from datetime import datetime, timedelta
        since_date = (datetime.now() - timedelta(days=days)).strftime('%d-%b-%Y')
        return since_date
    
    def _manual_filter_by_sender(self, mail: imaplib.IMAP4_SSL, email_ids: List[bytes], 
                                target_senders: List[str], since_days: int) -> List[bytes]:
        """Manually filter emails by sender and date"""
        import re
        from datetime import datetime, timedelta
        from email.utils import parsedate_to_datetime
        
        filtered_ids = []
        cutoff_date = datetime.now() - timedelta(days=since_days)
        
        for email_id in email_ids:
            try:
                # Fetch header only for efficiency
                status, msg_data = mail.fetch(email_id, '(RFC822.HEADER)')
                if status == 'OK':
                    header_data = msg_data[0][1].decode('utf-8', errors='ignore')
                    
                    # Extract sender from header
                    sender_match = re.search(r'From:\s*.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?', header_data, re.IGNORECASE)
                    sender = sender_match.group(1).strip() if sender_match else ""
                    
                    # Check if sender matches
                    if self._is_sender_match(sender, target_senders):
                        # Check date
                        date_match = re.search(r'Date:\s*(.+?)\r?\n', header_data, re.IGNORECASE)
                        if date_match:
                            try:
                                email_date = parsedate_to_datetime(date_match.group(1))
                                if email_date >= cutoff_date:
                                    filtered_ids.append(email_id)
                            except:
                                # Include if date parsing fails
                                filtered_ids.append(email_id)
                        else:
                            filtered_ids.append(email_id)
            
            except:
                continue
        
        return filtered_ids
    
    def _is_sender_match(self, sender: str, target_senders: List[str]) -> bool:
        """Check if sender matches any target sender (case insensitive contains)"""
        if not sender or not target_senders:
            return False
        
        sender_lower = sender.lower().strip()
        
        for target_sender in target_senders:
            target_lower = target_sender.lower().strip()
            if target_lower in sender_lower or sender_lower in target_lower:
                return True
        
        return False