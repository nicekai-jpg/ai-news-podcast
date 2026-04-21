import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import format_datetime
from typing import Any


def build_feed_xml(
    *,
    base_url: str,
    podcast_title: str,
    podcast_description: str,
    podcast_language: str,
    podcast_author: str,
    podcast_category: str,
    podcast_explicit: bool,
    episodes: list[dict[str, Any]],
) -> str:
    ns_atom = "http://www.w3.org/2005/Atom"
    ns_itunes = "http://www.itunes.com/dtds/podcast-1.0.dtd"
    ET.register_namespace("atom", ns_atom)
    ET.register_namespace("itunes", ns_itunes)

    rss = ET.Element("rss", {"version": "2.0"})
    channel = ET.SubElement(rss, "channel")

    ET.SubElement(channel, "title").text = podcast_title
    ET.SubElement(channel, "link").text = base_url + "/"
    ET.SubElement(channel, "description").text = podcast_description
    ET.SubElement(channel, "language").text = podcast_language
    ET.SubElement(channel, "lastBuildDate").text = format_datetime(datetime.now(tz=timezone.utc))
    ET.SubElement(channel, f"{{{ns_itunes}}}author").text = podcast_author
    ET.SubElement(channel, f"{{{ns_itunes}}}explicit").text = "yes" if podcast_explicit else "no"
    ET.SubElement(channel, f"{{{ns_itunes}}}category", {"text": podcast_category})
    ET.SubElement(channel, f"{{{ns_itunes}}}image", {"href": base_url + "/logo.png"})
    ET.SubElement(
        channel,
        f"{{{ns_atom}}}link",
        {
            "href": base_url + "/feed.xml",
            "rel": "self",
            "type": "application/rss+xml",
        },
    )

    for ep in episodes:
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = str(ep["title"])
        ET.SubElement(item, "description").text = str(ep["description"])
        ET.SubElement(item, "pubDate").text = str(ep["pubDate"])
        guid = ET.SubElement(item, "guid", {"isPermaLink": "false"})
        guid.text = str(ep["guid"])
        ET.SubElement(item, "link").text = str(ep["link"])
        ET.SubElement(
            item,
            "enclosure",
            {
                "url": str(ep["enclosure_url"]),
                "length": str(ep["enclosure_length"]),
                "type": "audio/mpeg",
            },
        )

    xml_bytes = ET.tostring(rss, encoding="utf-8", xml_declaration=True)
    return xml_bytes.decode("utf-8") + "\n"
