from typing_extensions import List
import requests
from xml.etree import ElementTree

def get_docs_urls(site_url:str) -> List[str]:
    """Get URLs docs from sitemap.xml"""
    try:
        response = requests.get(site_url)
        response.raise_for_status()

        # Parse the XML
        root = ElementTree.fromstring(response.content)

        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []
