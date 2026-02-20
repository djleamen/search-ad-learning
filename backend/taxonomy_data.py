"""
This module defines the taxonomy of categories and tags, as well as functions 
for tokenizing queries, building a seed corpus for training, and expanding 
queries into relevant tags based on predicted category probabilities.
"""

from __future__ import annotations

from hashlib import blake2b
import math
from typing import Dict, List, Tuple

TAXONOMY: Dict[str, Dict[str, List[str]]] = {
    "/Arts & Entertainment": {
        "tags": ["indie film", "festival tickets", "stand-up comedy", "live concert", "streaming series", "retro arcade", "celebrity news", "movie trailers", "music production", "fan conventions"],
        "training_queries": ["best comedy movies", "new music releases", "gaming stream highlights", "concerts near me", "funny podcast clips", "film festival schedule", "movie soundtrack vinyl", "stand up tour dates", "fan expo", "comic convention tickets"],
    },
    "/Autos & Vehicles": {
        "tags": ["electric suv", "truck accessories", "motorcycle helmets", "fleet leasing", "cargo vans", "off-road tires", "car detailing", "vehicle financing", "used pickup", "diesel maintenance"],
        "training_queries": ["best suv for families", "truck towing capacity", "motorcycle gear shop", "commercial van lease", "compare hybrid cars", "off road wheel kits", "auto detailing service", "fleet vehicle insurance"],
    },
    "/Beauty & Fitness": {
        "tags": ["hydrating serum", "hair growth kit", "organic makeup", "spa packages", "pilates studio", "yoga retreat", "protein isolate", "bodybuilding split", "skin clinic", "lash extensions"],
        "training_queries": ["best skincare routine", "hair care products", "yoga classes nearby", "bodybuilding workout plan", "spa day deals", "clean beauty brands", "protein powder review", "pilates reformer classes"],
    },
    "/Books & Literature": {
        "tags": ["ebook subscriptions", "poetry anthology", "children classics", "mystery novels", "book club picks", "indie magazines", "author interviews", "graphic novels", "audiobook bundles", "short story collection"],
        "training_queries": ["best mystery novels", "ebooks for kids", "poetry books to read", "literary magazine subscription", "book club recommendations", "audiobook app", "new fantasy author", "children literature list"],
    },
    "/Business & Industrial": {
        "tags": ["b2b lead gen", "trade show booth", "supply chain tools", "manufacturing erp", "industrial textiles", "warehouse automation", "corporate event planner", "market research", "logistics platform", "branding agency"],
        "training_queries": ["advertising agency services", "corporate event venue", "logistics software", "manufacturing consulting", "industrial textile suppliers", "warehouse management tools", "b2b marketing strategy", "trade show design"],
    },
    "/Computers & Electronics": {
        "tags": ["custom pc build", "mesh wifi", "gaming monitor", "noise cancelling earbuds", "dev laptop", "network security", "ssd upgrade", "smart home hub", "graphics card", "usb-c docking"],
        "training_queries": ["best laptop for coding", "pc hardware deals", "wifi router comparison", "consumer electronics sale", "network troubleshooting", "smart home setup", "ssd for gaming", "new graphics card", "python code", "software engineering tutorials", "fix keyboard issue", "ceo of microsoft"],
    },
    "/Finance": {
        "tags": ["high-yield savings", "student loan refinance", "retirement calculator", "crypto portfolio", "credit monitoring", "small business banking", "term life insurance", "mortgage rates", "index fund strategy", "tax optimization"],
        "training_queries": ["best investing apps", "bank account bonus", "insurance policy comparison", "loan interest rates", "how to build credit", "mortgage payment calculator", "retirement planning", "tax planning tips"],
    },
    "/Food & Drink": {
        "tags": ["meal prep ideas", "specialty coffee", "vegan recipes", "fine dining", "mocktail recipes", "air fryer meals", "craft beverages", "food delivery", "kitchen gadgets", "dessert baking"],
        "training_queries": ["easy dinner recipes", "best coffee beans", "restaurants near me", "healthy meal prep", "cocktail recipes", "baking ideas", "food delivery discounts", "kitchen tools review"],
    },
    "/Games": {
        "tags": ["battle royale", "mmorpg builds", "mobile gacha", "casino bonus", "esports streams", "speedrun guides", "coop shooters", "gaming skins", "indie platformer", "tournament brackets"],
        "training_queries": ["best online games", "video game tournaments", "casino app bonuses", "multiplayer shooter tips", "new indie games", "esports schedule", "game pass deals", "ranked ladder strategy"],
    },
    "/Health": {
        "tags": ["telehealth visits", "vitamin plans", "mental wellness", "heart health", "nutrition coaching", "sleep tracking", "injury rehab", "preventive care", "health screening", "meditation apps"],
        "training_queries": ["medical clinic near me", "wellness supplements", "healthy nutrition plan", "mental health resources", "fitness and wellness", "sleep improvement tips", "preventive checkups", "rehab exercises"],
    },
    "/Hobbies & Leisure": {
        "tags": ["camping gear", "trail maps", "drone photography", "collectible antiques", "woodworking tools", "fishing reels", "birdwatching lenses", "board game nights", "craft supplies", "pottery class"],
        "training_queries": ["camping checklist", "outdoor hobbies", "photography gear", "antique market", "woodworking projects", "fishing equipment", "pottery workshops", "board game groups"],
    },
    "/Home & Garden": {
        "tags": ["indoor plants", "kitchen remodel", "diy shelving", "smart thermostat", "home decor", "lawn care", "patio furniture", "paint palettes", "garden irrigation", "tile backsplash"],
        "training_queries": ["home improvement ideas", "garden planning guide", "interior design styles", "diy home projects", "best lawn tools", "kitchen renovation cost", "patio decor", "smart home upgrades"],
    },
    "/Internet & Telecom": {
        "tags": ["seo audits", "email campaigns", "social analytics", "domain hosting", "5g plans", "web design agency", "landing page optimization", "content calendar", "creator tools", "broadband deals"],
        "training_queries": ["web design services", "seo keyword tools", "email marketing platform", "social media strategy", "best mobile plans", "internet provider comparison", "content marketing", "domain registration"],
    },
    "/Jobs & Education": {
        "tags": ["resume templates", "interview prep", "online certifications", "coding bootcamp", "language courses", "career coaching", "study planner", "job board alerts", "leadership training", "internship programs"],
        "training_queries": ["career advice", "online education programs", "job interview tips", "professional training", "resume help", "certification courses", "how to switch careers", "internship opportunities", "ontario tech university", "university admissions", "college application deadlines"],
    },
    "/Law & Government": {
        "tags": ["legal consultation", "small claims", "immigration law", "policy updates", "public records", "civic engagement", "tax law", "compliance audits", "government grants", "election coverage"],
        "training_queries": ["find legal services", "public policy news", "government forms", "tax law questions", "business compliance help", "immigration attorney", "legal aid resources", "election information"],
    },
    "/Pets & Animals": {
        "tags": ["pet insurance", "dog training", "cat nutrition", "aquarium supplies", "adoption centers", "grooming service", "pet boarding", "veterinary clinic", "bird care", "animal rescue"],
        "training_queries": ["pet supplies near me", "best dog food", "cat grooming tips", "animal adoption", "veterinary services", "pet insurance plans", "dog training classes", "aquarium setup guide"],
    },
    "/Real Estate": {
        "tags": ["rental listings", "mortgage preapproval", "home appraisal", "commercial office space", "property management", "open house", "first-time buyer", "real estate agent", "condo market", "investment properties"],
        "training_queries": ["homes for sale", "rental apartments", "commercial real estate", "property investment tips", "mortgage preapproval", "find real estate agent", "housing market trends", "open house schedule"],
    },
    "/Shopping": {
        "tags": ["seasonal apparel", "flash deals", "luxury accessories", "price comparisons", "sneaker drops", "retail rewards", "electronics bundles", "coupon codes", "gift guides", "online marketplace"],
        "training_queries": ["online shopping deals", "best apparel brands", "accessories sale", "electronics discounts", "retail store near me", "coupon finder", "gift ideas", "compare product prices"],
    },
    "/Sports": {
        "tags": ["team jerseys", "league standings", "home gym setup", "marathon training", "basketball shoes", "soccer analytics", "fan tickets", "fitness tracker", "tennis racket", "sports nutrition"],
        "training_queries": ["sports team schedule", "league news", "fitness equipment", "best running shoes", "game tickets", "training for marathon", "home gym ideas", "sports supplements", "hockey canada", "nhl standings"],
    },
    "/Travel": {
        "tags": ["budget flights", "boutique hotels", "weekend getaways", "travel insurance", "visa checklists", "guided tours", "beach resorts", "city passes", "business travel", "adventure trips"],
        "training_queries": ["cheap flights", "hotel booking", "tourist destinations", "travel package deals", "best places to visit", "travel insurance plans", "weekend trip ideas", "international visa requirements", "things to do in toronto", "city sightseeing guide"],
    },
}

CATEGORY_LIST: List[str] = list(TAXONOMY.keys())

CATEGORY_HINTS: Dict[str, List[str]] = {
    "/Arts & Entertainment": ["movie", "movies", "music", "concert", "festival", "show", "comedy", "streaming"],
    "/Autos & Vehicles": ["car", "cars", "auto", "vehicle", "truck", "suv", "motorcycle", "leasing"],
    "/Beauty & Fitness": ["beauty", "cosmetics", "skincare", "hair", "spa", "yoga", "bodybuilding", "makeup"],
    "/Books & Literature": ["book", "books", "novel", "poetry", "magazine", "ebook", "audiobook", "literature"],
    "/Business & Industrial": ["b2b", "logistics", "manufacturing", "warehouse", "industrial", "marketing", "corporate", "trade"],
    "/Computers & Electronics": ["laptop", "pc", "computer", "hardware", "software", "network", "router", "electronics", "code", "coding", "python", "programming", "keyboard", "microsoft"],
    "/Finance": ["bank", "banking", "loan", "credit", "mortgage", "invest", "insurance", "tax"],
    "/Food & Drink": ["food", "drink", "recipe", "dining", "restaurant", "meal", "coffee", "beverage"],
    "/Games": ["game", "games", "gaming", "esports", "casino", "multiplayer", "ranked", "tournament"],
    "/Health": ["health", "medical", "wellness", "nutrition", "clinic", "rehab", "telehealth", "sleep"],
    "/Hobbies & Leisure": ["hobby", "camping", "outdoors", "photography", "antiques", "craft", "fishing", "pottery"],
    "/Home & Garden": ["home", "garden", "interior", "renovation", "diy", "lawn", "patio", "decor"],
    "/Internet & Telecom": ["seo", "web", "social", "email", "domain", "hosting", "telecom", "broadband"],
    "/Jobs & Education": ["jobs", "career", "resume", "interview", "education", "training", "course", "certification", "university", "college", "campus", "admissions"],
    "/Law & Government": ["law", "legal", "government", "policy", "compliance", "public", "attorney", "election"],
    "/Pets & Animals": ["pet", "pets", "dog", "cat", "veterinary", "adoption", "animal", "grooming"],
    "/Real Estate": ["real estate", "property", "house", "homes", "apartment", "rental", "realtor", "condo"],
    "/Shopping": ["shopping", "buy", "sale", "retail", "coupon", "apparel", "accessories", "marketplace"],
    "/Sports": ["sports", "team", "league", "fitness", "marathon", "jersey", "tickets", "training", "hockey", "nhl", "canada"],
    "/Travel": ["travel", "flight", "flights", "hotel", "hotels", "tour", "visa", "destination", "toronto", "sightseeing", "itinerary"],
}

CATEGORY_PHRASE_HINTS: Dict[str, List[str]] = {
    "/Sports": ["hockey canada", "nhl", "hockey"],
    "/Arts & Entertainment": ["fan expo", "comic con", "concert"],
    "/Travel": ["things to do in", "places to visit", "tourist attractions"],
    "/Jobs & Education": ["tech university", "university", "college", "online course"],
    "/Computers & Electronics": ["python code", "source code", "software", "computer", "keyboard", "microsoft"],
}


def tokenize(text: str) -> List[str]:
    """
    Tokenize the input text into a list of lowercase alphanumeric tokens.
    :param text: The input text to tokenize.
    :return: A list of tokens.
    """
    return [token for token in "".join(ch if ch.isalnum() else " " for ch in text.lower()).split() if len(token) > 1]


def normalize_token(token: str) -> str:
    """
    Normalize a token by applying basic stemming rules.
    :param token: The token to normalize.
    :return: The normalized token.
    """
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def tokenize_normalized(text: str) -> List[str]:
    """
    Tokenize and normalize the input text.
    :param text: The input text to tokenize and normalize.
    :return: A list of normalized tokens.
    """
    return [normalize_token(token) for token in tokenize(text)]


def _deterministic_noise(seed: str) -> float:
    """
    Generate a deterministic noise value between 0 and 1 based on the input seed string.
    :param seed: The input string to generate noise from.
    :return: A deterministic noise value between 0 and 1.
    """
    digest = blake2b(seed.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") / 2**32


def build_seed_corpus() -> List[Tuple[str, str]]:
    """
    Build a seed corpus for training by generating query-category pairs.
    :return: A list of tuples containing query and category pairs.
    """
    templates = [
        "best {}",
        "buy {} online",
        "top {} deals",
        "near me {}",
        "how to choose {}",
        "compare {} options",
        "reviews for {}",
        "{} for beginners",
        "{} recommendations",
    ]

    samples: List[Tuple[str, str]] = []
    for category, payload in TAXONOMY.items():
        for query in payload["training_queries"]:
            samples.append((query, category))

        for tag in payload["tags"]:
            samples.append((tag, category))
            for template in templates:
                samples.append((template.format(tag), category))

    return samples


def expand_query_to_tags(query: str, probabilities: Dict[str, float], top_k: int = 40) -> List[Tuple[str, str, float]]:
    """
    Expand a query into potential tags with associated categories and scores.
    :param query: The input query string.
    :param probabilities: A dictionary of category probabilities.
    :param top_k: The maximum number of tags to return.
    :return: A list of tuples containing tag, category, and score.
    """
    query_tokens = set(tokenize_normalized(query))
    scored: List[Tuple[str, str, float]] = []

    for category, probability in probabilities.items():
        if probability <= 0.01:
            continue

        for tag in TAXONOMY[category]["tags"]:
            tag_tokens = tokenize_normalized(tag)
            overlap = 0.0
            if tag_tokens:
                overlap = sum(1 for token in tag_tokens if token in query_tokens) / len(tag_tokens)

            diversity = 0.15 + 0.85 * _deterministic_noise(f"{query}:{category}:{tag}")
            score = probability * (0.45 + overlap * 0.9) * diversity
            scored.append((tag, category, score))

    scored.sort(key=lambda item: item[2], reverse=True)
    return scored[:top_k]


def lexical_category_probabilities(query: str) -> Dict[str, float]:
    """
    Compute lexical category probabilities for a given query based on token overlap and phrase hints.
    :param query: The input query string.
    :return: A dictionary mapping categories to their lexical probabilities.
    """
    query_tokens = set(tokenize_normalized(query))
    raw_scores: Dict[str, float] = {category: 0.01 for category in CATEGORY_LIST}

    for category in CATEGORY_LIST:
        payload = TAXONOMY[category]
        phrases = payload["tags"] + payload["training_queries"]

        for phrase in phrases:
            phrase_tokens = tokenize_normalized(phrase)
            if not phrase_tokens:
                continue

            overlap = sum(1 for token in phrase_tokens if token in query_tokens)
            if overlap:
                raw_scores[category] += overlap / len(phrase_tokens)

            if phrase in query.lower():
                raw_scores[category] += 0.45

        hint_tokens = [normalize_token(token) for token in CATEGORY_HINTS[category]]
        hint_overlap = sum(1 for token in hint_tokens if token in query_tokens)
        raw_scores[category] += hint_overlap * 0.55

        for phrase_hint in CATEGORY_PHRASE_HINTS.get(category, []):
            if phrase_hint in query.lower():
                raw_scores[category] += 1.25

    max_score = max(raw_scores.values())
    exp_scores = {category: math.exp(score - max_score) for category, score in raw_scores.items()}
    total = sum(exp_scores.values()) or 1.0
    return {category: value / total for category, value in exp_scores.items()}
