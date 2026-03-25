import os
import time
import re
import requests
import psycopg2
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

WP_BASE = os.getenv("WORDPRESS_BASE_URL", "").rstrip("/")
DB_URL = os.getenv("SUPABASE_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not WP_BASE or not DB_URL or not OPENAI_API_KEY:
    raise ValueError("Missing one or more required environment variables in .env")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=60,
    max_retries=5,
)


def get_db_connection():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    return conn


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200):
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)

        if end < n:
            while end > start and text[end - 1] not in [" ", ".", "\n"]:
                end -= 1

            if end <= start + 200:
                end = min(start + max_chars, n)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(start + 1, end - overlap)

    return chunks


def embed_text(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def fetch_all_posts(per_page=25, max_pages=100):
    all_posts = []

    for page in range(1, max_pages + 1):
        url = f"{WP_BASE}/wp-json/wp/v2/posts"

        for attempt in range(3):
            try:
                resp = requests.get(
                    url,
                    params={"per_page": per_page, "page": page},
                    timeout=60
                )

                if resp.status_code == 400:
                    return all_posts

                resp.raise_for_status()
                data = resp.json()

                if not data:
                    return all_posts

                all_posts.extend(data)
                print(f"Fetched page {page}, total posts so far: {len(all_posts)}")
                time.sleep(0.5)
                break

            except Exception as e:
                print(f"Error fetching page {page}, attempt {attempt + 1}/3: {e}")
                time.sleep(3)

                if attempt == 2:
                    print(f"Stopping at page {page} after repeated failures.")
                    return all_posts

    return all_posts


def get_existing_article_meta(cur, article_id: int):
    cur.execute("""
        select wp_modified_at
        from articles
        where id = %s
    """, (article_id,))
    return cur.fetchone()


def should_ingest(post, existing_row):
    wp_modified = post.get("modified")

    if existing_row is None:
        return True

    existing_modified = existing_row[0]

    if existing_modified is None:
        return True

    return str(existing_modified) != str(wp_modified)


def upsert_article(cur, post, cleaned_text: str):
    article_id = post["id"]
    slug = post.get("slug")
    title = post["title"]["rendered"]
    url = post["link"]
    raw_html = post["content"]["rendered"]
    published_at = post["date"]
    modified_at = post.get("modified")

    cur.execute("""
        insert into articles (
            id, slug, title, url, category, tags, author,
            published_at, wp_modified_at, raw_html, cleaned_text, last_ingested_at
        )
        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        on conflict (id) do update
        set slug = excluded.slug,
            title = excluded.title,
            url = excluded.url,
            published_at = excluded.published_at,
            wp_modified_at = excluded.wp_modified_at,
            raw_html = excluded.raw_html,
            cleaned_text = excluded.cleaned_text,
            last_ingested_at = now()
    """, (
        article_id,
        slug,
        title,
        url,
        None,
        [],
        None,
        published_at,
        modified_at,
        raw_html,
        cleaned_text
    ))

    return article_id


def delete_existing_chunks(cur, article_id: int):
    cur.execute("delete from article_chunks where article_id = %s", (article_id,))


def insert_chunks(cur, article_id: int, cleaned_text: str):
    chunks = chunk_text(cleaned_text)
    print(f"Article {article_id}: {len(chunks)} chunks")

    for idx, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        cur.execute("""
            insert into article_chunks (article_id, chunk_index, content, embedding)
            values (%s, %s, %s, %s)
        """, (article_id, idx, chunk, embedding))
        time.sleep(0.3)


def run_ingestion():
    posts = fetch_all_posts()
    print(f"Total posts fetched: {len(posts)}")

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            for post in posts:
                try:
                    article_id = post["id"]
                    existing = get_existing_article_meta(cur, article_id)

                    if not should_ingest(post, existing):
                        print(f"Skipping unchanged post {article_id}")
                        continue

                    cleaned_text = clean_html(post["content"]["rendered"])
                    if not cleaned_text.strip():
                        print(f"Skipping empty cleaned text for post {article_id}")
                        continue

                    upsert_article(cur, post, cleaned_text)
                    delete_existing_chunks(cur, article_id)
                    insert_chunks(cur, article_id, cleaned_text)

                    print(f"Ingested article {article_id}: {post['title']['rendered']}")

                except Exception as e:
                    print(f"Error processing post {post.get('id')}: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    run_ingestion()