"""
Email Thread-Aware Chunking Strategy.

The "Senior" Pattern:
1. Reply Stripping: Remove quoted reply text
2. Signature Removal: Strip email signatures
3. Context Injection: Prepend Subject, Sender, Timestamp
4. Granularity: Chunk by individual message, not thread
"""
import re
from typing import List, Dict, Any
from email import message_from_string
from email_reply_parser import EmailReplyParser
import html2text
from .base import ChunkerBase, Chunk


class EmailThreadAwareChunker(ChunkerBase):
    """
    Email-specific chunking strategy.
    
    Designed for supply chain email threads where context matters.
    Each email in a thread becomes a separate chunk with full context.
    """
    
    def __init__(self):
        """Initialize email chunker."""
        self.html_parser = html2text.HTML2Text()
        self.html_parser.ignore_links = False
        self.html_parser.ignore_images = True
        
        # Common signature patterns
        self.signature_patterns = [
            r'--\s*\n',  # -- separator
            r'Sent from my \w+',
            r'Best regards,?',
            r'Kind regards,?',
            r'Sincerely,?',
            r'Thanks,?',
            r'Cheers,?',
            r'\n_{3,}',  # Underscores
        ]
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Chunk email thread into individual messages.
        
        Args:
            text: Raw email text (can be EML format or plain text)
            metadata: Must include thread_id, subject, sender, timestamp
        """
        if metadata is None:
            metadata = {}
        
        # Parse email if it's in EML format
        if text.startswith('From:') or text.startswith('Subject:'):
            email_msg = message_from_string(text)
            subject = email_msg.get('Subject', 'No Subject')
            sender = email_msg.get('From', 'Unknown')
            date = email_msg.get('Date', '')
            
            # Get body
            if email_msg.is_multipart():
                for part in email_msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode()
                        break
                    elif part.get_content_type() == "text/html":
                        html_body = part.get_payload(decode=True).decode()
                        body = self.html_parser.handle(html_body)
                        break
                else:
                    body = text
            else:
                body = email_msg.get_payload(decode=True)
                if isinstance(body, bytes):
                    body = body.decode()
                else:
                    body = str(body)
        else:
            # Plain text email
            subject = metadata.get('subject', 'No Subject')
            sender = metadata.get('sender', 'Unknown')
            date = metadata.get('timestamp', '')
            body = text
        
        # Step 1: Strip quoted replies
        cleaned_body = EmailReplyParser.parse_reply(body)
        
        # Step 2: Remove signature
        cleaned_body = self._remove_signature(cleaned_body)
        
        # Step 3: Build context-injected chunk
        context_prefix = f"[Subject: {subject}] [From: {sender}]"
        if date:
            context_prefix += f" [Date: {date}]"
        
        full_chunk_text = f"{context_prefix}\n\n{cleaned_body.strip()}"
        
        # Create single chunk for this email
        chunk = Chunk(
            text=full_chunk_text,
            metadata={
                **metadata,
                "subject": subject,
                "sender": sender,
                "timestamp": date,
                "chunk_index": 0,
                "chunking_strategy": "email_thread_aware",
                "has_attachment": metadata.get("has_attachment", False)
            },
            chunk_index=0
        )
        
        return [chunk]
    
    def _remove_signature(self, text: str) -> str:
        """
        Remove email signature from text.
        
        Uses pattern matching for common signature formats.
        """
        lines = text.split('\n')
        
        # Find signature start
        sig_start_idx = len(lines)
        
        for i, line in enumerate(lines):
            # Check for signature patterns
            for pattern in self.signature_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    sig_start_idx = i
                    break
            
            # Check for common signature indicators
            if i > 0 and len(line.strip()) > 0:
                # Short lines at the end often indicate signatures
                if len(line) < 50 and (
                    'regard' in line.lower() or
                    'thank' in line.lower() or
                    'best' in line.lower() or
                    'sent from' in line.lower()
                ):
                    sig_start_idx = i
                    break
        
        # Remove signature
        cleaned_lines = lines[:sig_start_idx]
        return '\n'.join(cleaned_lines).strip()
    
    def get_info(self) -> Dict[str, Any]:
        """Return information about this chunking strategy."""
        return {
            "name": "Email Thread-Aware Chunker",
            "description": "Cleans emails (strips replies & signatures), adds context (subject/sender)",
            "features": [
                "Reply stripping using email-reply-parser",
                "Signature removal via pattern matching",
                "Context injection (subject, sender, date)",
                "HTML email support"
            ],
            "best_for": "Supply chain email threads, operational communication"
        }

