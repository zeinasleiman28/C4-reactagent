"""
tools.py
--------
The tools the Bridgeway agent can call:

    search_business_info(query)                      -> looks up facts in the business documents (C4)
    record_customer_interest(email, name, message)  -> saves a lead (an NGO interested in a service)
    record_feedback(question)                        -> saves feedback or a question we couldn't answer
    estimate_campaign_package(ngo_size, goal)        -> price estimate from business_data.py (C4)

Each tool:
1. validates its input and returns {"status": "error", "reason": ...} if something is wrong,
   so the model can ask the user to fix it (instead of the program crashing);
2. prints a short log line (you see it in the notebook / terminal);
3. appends one JSON line to a file in logs/ (leads.jsonl or feedback.jsonl);
4. returns {"status": "recorded", "id": ...} on success.

The tools only *record* things. All business facts and prices stay in business_data.py.

This file lives outside the notebook so the tests in tests/ can import it.
"""

import json
import re
import uuid

# Shared helpers: timestamps, JSONL files in logs/, emoji-safe printing.
# Tests can point the logs somewhere else: log_utils.LOG_DIR = tmp_path
from log_utils import append_jsonl, now_iso, safe_print

LEADS_FILE = "leads.jsonl"
FEEDBACK_FILE = "feedback.jsonl"

# Simple "looks like an email" check: something@something.something (no spaces).
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Upper limits so a runaway model can't write huge entries into our logs.
MAX_NAME_LENGTH = 100
MAX_TEXT_LENGTH = 2000


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _new_id(prefix):
    """A short unique id such as 'lead-3f9a1c2b'."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _error(reason):
    """The standard error result every tool returns."""
    return {"status": "error", "reason": reason}


def _clean_text(value):
    """Turn the value into a stripped string ('' for None)."""
    if value is None:
        return ""
    return str(value).strip()


# ---------------------------------------------------------------------------
# Tool 1: record_customer_interest
# ---------------------------------------------------------------------------

def record_customer_interest(email, name, message=""):
    """Save a potential customer's contact details and what they are interested in."""
    email = _clean_text(email)
    name = _clean_text(name)
    message = _clean_text(message)

    # Validate. Each error says exactly what the model should ask the user to fix.
    if not name:
        return _error("Missing name. Please ask the user for their name.")
    if len(name) > MAX_NAME_LENGTH:
        return _error(f"Name is too long (max {MAX_NAME_LENGTH} characters).")
    if not email:
        return _error("Missing email. Please ask the user for their email address.")
    if not EMAIL_PATTERN.match(email):
        return _error(f"'{email}' does not look like a valid email. Please ask the user to check it.")
    if len(message) > MAX_TEXT_LENGTH:
        return _error(f"Message is too long (max {MAX_TEXT_LENGTH} characters). Please summarize it.")

    record = {
        "id": _new_id("lead"),
        "timestamp": now_iso(),
        "name": name,
        "email": email,
        "message": message,
    }
    try:
        append_jsonl(LEADS_FILE, record)
    except OSError as e:
        return _error(f"Could not save the lead ({e}). Ask the user to email us directly.")

    safe_print(f"📝 [LEAD] {name} <{email}>: {message or '(no details)'}")
    return {"status": "recorded", "id": record["id"]}


# ---------------------------------------------------------------------------
# Tool 2: record_feedback
# ---------------------------------------------------------------------------

def record_feedback(question):
    """Save user feedback, or a question the assistant could not answer."""
    question = _clean_text(question)

    if not question:
        return _error("Missing text. Pass the user's feedback or unanswered question.")
    if len(question) > MAX_TEXT_LENGTH:
        return _error(f"Text is too long (max {MAX_TEXT_LENGTH} characters). Please summarize it.")

    record = {
        "id": _new_id("fb"),
        "timestamp": now_iso(),
        "question": question,
    }
    try:
        append_jsonl(FEEDBACK_FILE, record)
    except OSError as e:
        return _error(f"Could not save the feedback ({e}).")

    safe_print(f"💬 [FEEDBACK] {question}")
    return {"status": "recorded", "id": record["id"]}


# ---------------------------------------------------------------------------
# Tool 0 (C4): search_business_info
# ---------------------------------------------------------------------------
# In C3 the whole business text was pasted into the system prompt. In C4 the agent
# must *look things up* instead: this tool is a tiny keyword search over the
# business summary + the PDF, so the model has to decide WHEN to search (ReAct).

_FACTS = None  # loaded once, on first use

# Words too common to be useful for matching ("what is the price" -> only "price" counts)
_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "can", "do", "does", "for", "from", "how", "i", "in", "is", "it",
    "me", "my", "of", "on", "or", "our", "tell", "the", "to", "us", "we", "what", "which", "who", "with",
    "you", "your", "about", "much", "many", "there", "any", "have", "has", "will", "would", "like",
}
MAX_SEARCH_RESULTS = 5


def _load_facts():
    """Split the business documents into one 'fact' per line/sentence (cached)."""
    global _FACTS
    if _FACTS is None:
        from knowledge import load_business_profile, load_business_summary  # imported here: avoids a cycle
        facts = []
        for paragraph in load_business_summary().splitlines():
            facts.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip())
        facts.extend(line for line in load_business_profile().splitlines() if len(line) > 15)
        # Drop control characters that PDF bullets leave behind, and empty leftovers.
        _FACTS = [f for f in (re.sub(r"[\x00-\x1f\x7f]", "", fact).strip() for fact in facts) if f]
    return _FACTS


def _keywords(text):
    """Lower-case words of 3+ letters/digits that are not stopwords."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3 and w not in _STOPWORDS}


def _similar(a, b):
    """True when two words are the same word in a different form (plural, verb form, ...)."""
    return a == b or a.startswith(b) or b.startswith(a) or (len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5])


def search_business_info(query):
    """Return the business facts that best match the query (keyword overlap)."""
    query = _clean_text(query)
    if not query:
        return _error("Missing query. Pass a short search phrase, e.g. 'donor campaign price'.")
    words = _keywords(query)
    if not words:
        return _error("The query has no searchable words. Try specific words such as 'volunteer drive price'.")

    scored = []
    for fact in _load_facts():
        fact_words = _keywords(fact)
        # Count query words that match a fact word exactly, as a prefix ("volunteer" ~ "volunteers")
        # or by a shared 5-letter stem ("founded" ~ "founder", "partners" ~ "partnership").
        hits = sum(1 for w in words if any(_similar(w, fw) for fw in fact_words))
        if hits:
            scored.append((hits, fact))
    scored.sort(key=lambda pair: -pair[0])

    safe_print(f"🔎 [SEARCH] '{query}' -> {len(scored)} matching fact(s)")
    if not scored:
        return {"status": "no_match", "results": [], "note": "Nothing in the business documents matches this query."}
    return {"status": "ok", "results": [fact for _, fact in scored[:MAX_SEARCH_RESULTS]]}


# ---------------------------------------------------------------------------
# Tool 3 (C4): estimate_campaign_package
# ---------------------------------------------------------------------------
# The model never does arithmetic itself: prices and discount rules come from
# business_data.py, which holds exactly the numbers printed in the business documents.

from business_data import LABELS, PRICES, best_discount_percent, discounted_price

# Words in the user's goal -> the service they correspond to
GOAL_KEYWORDS = {
    "donor_outreach_campaign": ("donor", "donation", "fundrais", "funding"),
    "partnership_pitch_pack": ("partner", "sponsor", "corporate", "compan", "collaborat"),
    "volunteer_drive": ("volunteer",),
    "grant_deadline_tracking_per_month": ("grant", "deadline"),
    "translation_arabic_french": ("translat", "arabic", "french"),
    "extra_follow_up_wave": ("follow-up", "follow up", "followup", "extra wave", "reminder"),
}
# Accepted spellings of the NGO size -> "small" (budget < 100,000 USD) or "large"
SIZE_ALIASES = {"small": "small", "tiny": "small", "little": "small",
                "large": "large", "big": "large", "medium": "large", "mid": "large", "standard": "large"}


def estimate_campaign_package(ngo_size, goal):
    """Price estimate for the services matching `goal`, with the single applicable discount."""
    size_text = _clean_text(ngo_size).lower()
    goal_text = _clean_text(goal).lower()

    size = next((value for word, value in SIZE_ALIASES.items() if word in size_text), None)
    if size is None:
        return _error("ngo_size must be 'small' (annual budget under 100,000 USD) or 'large' "
                      "(100,000 USD or more). Ask the user which one applies.")
    if not goal_text:
        return _error("Missing goal. Say what the NGO wants: donors, partners, volunteers, grant tracking "
                      "(add-ons: translation, extra follow-up wave).")

    wanted = [key for key, words in GOAL_KEYWORDS.items() if any(w in goal_text for w in words)]
    if not wanted:
        return _error(f"Goal '{goal}' does not match a Bridgeway service. Options: donors (donor outreach "
                      "campaign), partners (partnership pitch pack), volunteers (volunteer drive), grants "
                      "(grant-deadline tracking); add-ons: translation, extra follow-up wave.")

    campaigns = [k for k in wanted if k in PRICES["campaigns"]]
    add_ons = [k for k in wanted if k in PRICES["add_ons"]]
    monthly = "grant_deadline_tracking_per_month" in wanted
    if add_ons and not campaigns:
        return _error("Add-ons (translation, extra follow-up wave) can only be added to a campaign. "
                      "Ask which campaign the NGO wants.")

    small = size == "small"
    services_booked = len(campaigns) + (1 if monthly else 0)     # add-ons do not count as services
    percent = best_discount_percent(small_ngo=small, services_booked=services_booked)

    items, one_time = [], 0
    for key in campaigns:
        items.append({"service": LABELS[key], "price_usd": PRICES["campaigns"][key], "billing": "per campaign"})
        one_time += PRICES["campaigns"][key]
    for key in add_ons:
        items.append({"service": LABELS[key], "price_usd": PRICES["add_ons"][key], "billing": "add-on"})
        one_time += PRICES["add_ons"][key]
    monthly_price = PRICES["grant_deadline_tracking_per_month"] if monthly else 0
    if monthly:
        items.append({"service": LABELS["grant_deadline_tracking_per_month"], "price_usd": monthly_price,
                      "billing": "per month"})

    if percent == 20:
        reason = "20% small-NGO discount (annual budget under 100,000 USD)"
        if services_booked >= 2:
            reason += "; the 10% bundle discount also applies but discounts never stack, so the bigger one is used"
    elif percent == 10:
        reason = "10% bundle discount (2 or more services booked together)"
    else:
        reason = "no discount (large NGO, single service)"

    result = {
        "status": "ok",
        "ngo_size": size,
        "items": items,
        "services_booked": services_booked,
        "discount_percent": percent,
        "discount_reason": reason,
        "one_time_subtotal_usd": one_time,
        "one_time_total_usd": discounted_price(one_time, small, services_booked),
        "monthly_total_usd": discounted_price(monthly_price, small, services_booked) if monthly else 0,
        "note": "Estimate only; the free 30-minute discovery call confirms the final scope. "
                "Bridgeway never promises how much money a campaign will raise.",
    }
    safe_print(f"🧮 [ESTIMATE] {size} NGO, goal '{goal}' -> {percent}% off, one-time {result['one_time_total_usd']} USD"
               + (f", {result['monthly_total_usd']} USD/month" if monthly else ""))
    return result


# ---------------------------------------------------------------------------
# Tool definitions, MCP-style
# ---------------------------------------------------------------------------
# MCP (Model Context Protocol) describes every tool with three fields:
#   name         - what the model calls
#   description  - when and why to use it (the model reads this!)
#   inputSchema  - a JSON Schema for the arguments
# We write the tools once in this format and convert them for the OpenAI API below.

MCP_TOOL_DEFINITIONS = [
    {
        "name": "search_business_info",
        "description": (
            "Search Bridgeway's business documents (services, prices, discounts, team, hours, contact, "
            "process, what is NOT offered). Call it BEFORE answering any factual question about the "
            "business; never answer prices or facts from memory. Use 2-5 specific keywords such as "
            "'donor outreach campaign price' or 'small NGO discount'. Returns {status: 'ok', results: [facts]} "
            "or {status: 'no_match'} when the documents say nothing about the topic."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short keyword search phrase, e.g. 'volunteer drive price'",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "record_customer_interest",
        "description": (
            "Record a lead: someone from an NGO who is interested in one of Bridgeway's services "
            "(donor outreach campaign, partnership pitch pack, volunteer drive, grant-deadline tracking) "
            "so the team can follow up by email and schedule the free discovery call. Call it as soon as "
            "you have at least the person's name and a valid-looking email; do not wait for anything else. "
            "Do not call it for general questions without contact details. Put the NGO name, the service "
            "they want and the timing in 'message'. Returns {status: 'recorded', id} or {status: 'error', reason}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The person's email address, e.g. maya@greensteps.org",
                },
                "name": {
                    "type": "string",
                    "description": "The person's name, e.g. Maya",
                },
                "message": {
                    "type": "string",
                    "description": "What they want: NGO name, service(s), timing. "
                                   "E.g. 'GreenSteps – volunteer drive wanted for March'",
                },
            },
            "required": ["email", "name", "message"],
        },
    },
    {
        "name": "record_feedback",
        "description": (
            "Record something the team must review. Call it when (a) the user asks about something "
            "not in the business information, (b) the user asks for a service Bridgeway does not offer "
            "(full grant proposals, buying/selling email lists, paid ads, handling donation money, "
            "outreach for political parties or for-profit companies), or (c) the user gives feedback, "
            "praise or a complaint. Returns {status: 'recorded', id} or {status: 'error', reason}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The user's unanswered question or their feedback, in their words.",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "estimate_campaign_package",
        "description": (
            "Compute a price estimate for Bridgeway services from the official price list and discount rules "
            "(20% off for small NGOs with an annual budget under 100,000 USD; 10% off for 2+ services; the two "
            "never stack). Call it whenever the user asks what something costs, for a quote, or for a budget. "
            "If you do not know whether the NGO is small or large, ask the user first. Returns the items, the "
            "discount applied and the totals, or {status: 'error', reason}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "ngo_size": {
                    "type": "string",
                    "enum": ["small", "large"],
                    "description": "'small' = annual budget under 100,000 USD, 'large' = 100,000 USD or more",
                },
                "goal": {
                    "type": "string",
                    "description": "What the NGO wants, in a few words: 'reach donors', 'find partners', "
                                   "'recruit volunteers', 'track grant deadlines'. Combine with 'and' for several "
                                   "services; mention 'translation' or 'extra follow-up wave' for add-ons.",
                },
            },
            "required": ["ngo_size", "goal"],
        },
    },
]


def to_openai_tools(mcp_definitions):
    """Convert MCP-style definitions into the 'tools' format of the OpenAI chat API."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"],
            },
        }
        for tool in mcp_definitions
    ]


# What we pass as tools=... to client.chat.completions.create()
OPENAI_TOOLS = to_openai_tools(MCP_TOOL_DEFINITIONS)

# Tool name (as the model says it) -> the Python function that runs it
TOOL_REGISTRY = {
    "search_business_info": search_business_info,
    "record_customer_interest": record_customer_interest,
    "record_feedback": record_feedback,
    "estimate_campaign_package": estimate_campaign_package,
}


# ---------------------------------------------------------------------------
# Running a tool call that came from the model
# ---------------------------------------------------------------------------

def run_tool(name, arguments):
    """
    Run the tool the model asked for and always return a dict (never raise).

    `arguments` is what the model sent: usually a JSON string like '{"question": "..."}',
    but a dict is accepted too. Unknown tools, broken JSON and wrong arguments all become
    {"status": "error", ...} so the model can recover.
    """
    func = TOOL_REGISTRY.get(name)
    if func is None:
        return _error(f"Unknown tool '{name}'. Available tools: {', '.join(TOOL_REGISTRY)}.")

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return _error(f"Arguments for '{name}' are not valid JSON: {arguments!r}")
    if not isinstance(arguments, dict):
        return _error(f"Arguments for '{name}' must be a JSON object.")

    # Ignore any extra arguments the model invented; pass only the ones the tool accepts.
    schema = next(t["inputSchema"] for t in MCP_TOOL_DEFINITIONS if t["name"] == name)
    allowed = schema["properties"].keys()
    kwargs = {key: value for key, value in arguments.items() if key in allowed}

    try:
        return func(**kwargs)
    except TypeError as e:  # e.g. a required argument is missing
        return _error(f"Wrong arguments for '{name}': {e}")
