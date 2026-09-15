"""Synthetic customer-support agent workload.

Each conversation is a system prompt with a product knowledge base (~2.5k tokens, shared across requests for
the same product, like a real RAG agent), tool definitions, and a customer question. Generation is seeded, so
every endpoint and every run sees byte-identical requests.
"""
import random

PRODUCTS = {
    "Ledgerly": "accounting software for small businesses",
    "ShipFast": "a shipping and fulfilment platform for online stores",
    "Clinico": "practice management software for clinics",
    "Brightdesk": "a help desk for B2B SaaS teams",
}
PLANS = ["Starter", "Growth", "Scale", "Enterprise"]
FEATURES = ["SSO", "audit logs", "API access", "custom roles", "data export", "multi-currency", "webhooks",
            "sandbox environments", "priority support", "usage analytics", "approval workflows", "IP allowlisting"]
INTEGRATIONS = ["Salesforce", "HubSpot", "Zendesk", "Slack", "Xero", "QuickBooks", "Shopify", "Stripe", "Intercom", "Microsoft Teams"]
ACTIONS = ["reset a password", "add a teammate", "change the billing email", "export invoices", "rotate an API key",
           "cancel a subscription", "set up SSO", "restore a deleted record", "change the account owner", "enable two-factor authentication"]
ROLES = ["an Owner", "an Admin", "any member with the Manage Workspace permission", "a Billing Admin"]
STEPS = [
    "Open Settings from the left sidebar.",
    "Select Workspace, then open the Security tab.",
    "Click the name of the member you want to update.",
    "Choose Edit and review the fields shown in the panel.",
    "Confirm the change by typing the workspace name.",
    "Click Save. A confirmation email is sent to every Owner.",
    "Go to Billing and select Payment details.",
    "Open the Integrations page and find the app in the directory.",
    "Click Connect and sign in with an administrator account.",
    "Map the fields you want to sync and choose a default owner.",
    "Download the CSV from the Exports page once the job completes.",
    "Refresh the page; the new setting appears under Active policies.",
]
ERRORS = ["E1042", "E2210", "E3007", "SYNC-409", "AUTH-401", "RATE-429", "IMPORT-422", "WEBHOOK-500"]

TOOLS = [
    {"type": "function", "function": {
        "name": "search_knowledge_base",
        "description": "Search the product knowledge base for articles relevant to a query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "search terms"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "create_ticket",
        "description": "Create a support ticket for a human agent when the knowledge base does not answer the question.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "normal", "urgent"]},
        }, "required": ["summary", "priority"]},
    }},
    {"type": "function", "function": {
        "name": "escalate_to_human",
        "description": "Hand the conversation to a human immediately. Use for billing disputes, security incidents or angry customers.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
    }},
]


def _howto(rng: random.Random, product: str, action: str) -> dict:
    plan = rng.choice(PLANS)
    body = [
        f"Only {rng.choice(ROLES)} can {action}. If you do not see the option, ask an administrator to check your role.",
        "Steps:",
        *[f"{k + 1}. {step}" for k, step in enumerate(rng.sample(STEPS, 5))],
        f"Changes take effect within {rng.choice([1, 5, 15, 30])} minutes. On the {plan} plan this setting can be locked by an "
        f"administrator, in which case the button is greyed out and shows a padlock icon.",
        "This action is recorded in the audit log together with the IP address of the person who made it.",
    ]
    return {"title": f"How to {action} in {product}", "body": "\n".join(body), "kind": "howto", "meta": {"action": action}}


def _plans(rng: random.Random, product: str) -> dict:
    a, b = rng.sample(PLANS, 2)
    body = [f"{product} is available on four plans. Prices are per workspace per month, billed annually; monthly billing costs 20% more."]
    for plan in PLANS:
        features = ", ".join(rng.sample(FEATURES, 4))
        body.append(f"- {plan}: ${rng.choice([19, 29, 49, 79, 99, 149, 299])}/month, includes {rng.choice([3, 5, 10, 25, 50, 100])} seats, "
                    f"additional seats ${rng.choice([5, 8, 12])} each. Includes {features}.")
    body.append("Downgrades take effect at the end of the current billing period. Upgrades are prorated immediately.")
    return {"title": f"{product} plans and pricing", "body": "\n".join(body), "kind": "plan", "meta": {"a": a, "b": b}}


def _integration(rng: random.Random, product: str, integration: str) -> dict:
    interval = rng.choice([5, 15, 30, 60])
    body = [
        f"The {integration} integration is available on the {rng.choice(PLANS[1:])} plan and above. It syncs every {interval} minutes; "
        f"you can trigger a manual sync from the integration page at most {rng.choice([3, 5, 10])} times per hour.",
        "Setup:",
        *[f"{k + 1}. {step}" for k, step in enumerate(rng.sample(STEPS, 4))],
        f"Known limitations: custom objects are not synced, attachments larger than {rng.choice([10, 25, 50])} MB are skipped, "
        f"and deleting a record in {integration} archives it in {product} instead of deleting it.",
    ]
    return {"title": f"Connecting {product} to {integration}", "body": "\n".join(body), "kind": "integration", "meta": {"integration": integration}}


def _policy(rng: random.Random, product: str) -> dict:
    body = [
        f"You can cancel at any time from Billing. Annual plans cancelled within {rng.choice([14, 30, 45])} days of purchase are refunded in full; "
        f"after that, no partial refunds are issued for the remaining term.",
        f"After cancellation your data is kept read-only for {rng.choice([30, 60, 90])} days and then permanently deleted. "
        f"Exports remain available during that period.",
        f"Support response targets: Starter {rng.choice([48, 72])} hours, Growth {rng.choice([24, 36])} hours, "
        f"Scale {rng.choice([8, 12])} hours, Enterprise {rng.choice([1, 2, 4])} hours with a named account manager.",
        "Billing disputes and chargebacks are handled by the finance team, not by support agents.",
    ]
    return {"title": f"{product} refund, cancellation and data retention policy", "body": "\n".join(body), "kind": "policy", "meta": {}}


def _troubleshooting(rng: random.Random, product: str, code: str) -> dict:
    cause = rng.choice([
        "an expired OAuth token for a connected app",
        "a field mapping that points at a deleted custom field",
        "too many requests from the same API key in a short period",
        "an import file with a header row that does not match the template",
    ])
    body = [
        f"Error {code} means {cause}.",
        "To fix it:",
        *[f"{k + 1}. {step}" for k, step in enumerate(rng.sample(STEPS, 3))],
        f"If the error persists for more than {rng.choice([1, 2, 24])} hours, contact support with the request ID shown under the error message.",
    ]
    return {"title": f"Troubleshooting error {code} in {product}", "body": "\n".join(body), "kind": "troubleshooting", "meta": {"code": code}}


def knowledge_base(product: str, num_articles: int = 14) -> list[dict]:
    # one plans article, one policy article, then distinct how-tos, integrations and error guides
    rng = random.Random(f"kb:{product}")
    actions = rng.sample(ACTIONS, len(ACTIONS))
    integrations = rng.sample(INTEGRATIONS, len(INTEGRATIONS))
    errors = rng.sample(ERRORS, len(ERRORS))
    articles = [_plans(rng, product), _policy(rng, product)]
    i = 0
    while len(articles) < num_articles:
        k = i // 3
        if i % 3 == 0:
            articles.append(_howto(rng, product, actions[k % len(actions)]))
        elif i % 3 == 1:
            articles.append(_integration(rng, product, integrations[k % len(integrations)]))
        else:
            articles.append(_troubleshooting(rng, product, errors[k % len(errors)]))
        i += 1
    return articles


def system_prompt(product: str, num_articles: int = 14) -> str:
    articles = "\n\n".join(f"## {a['title']}\n{a['body']}" for a in knowledge_base(product, num_articles))
    return (
        f"You are the customer support agent for {product}, {PRODUCTS[product]}. Answer the customer's question using only "
        f"the knowledge base articles below and name the article you used. If the articles do not contain the answer, "
        f"call create_ticket. If the customer reports a security incident or a billing dispute, call escalate_to_human. "
        f"Be concise and specific.\n\n# Knowledge base\n\n{articles}"
    )


def _question(rng: random.Random, product: str, articles: list[dict]) -> str:
    roll = rng.random()
    if roll < 0.1:
        return rng.choice([
            f"Does {product} support on-premise deployment in an air-gapped network?",
            f"Can {product} store our data in a specific region, like Frankfurt?",
            "Is there a discount for registered non-profits?",
        ])
    if roll < 0.18:
        return rng.choice([
            "Someone logged into our account from another country last night and changed our payout details. What do we do?",
            "We were charged twice this month and I want the second charge reversed immediately.",
        ])
    article = rng.choice(articles)
    meta = article["meta"]
    return {
        "howto": f"How do I {meta.get('action')}? We're on the {rng.choice(PLANS)} plan and I'm not sure I have permission.",
        "plan": f"What's the difference between the {meta.get('a')} and {meta.get('b')} plans, and what does an extra seat cost on {meta.get('a')}?",
        "integration": f"Does {product} work with {meta.get('integration')}? How often does it sync and what doesn't get synced?",
        "policy": f"If we cancel our annual plan after {rng.choice([10, 20, 40])} days, do we get a refund? What happens to our data?",
        "troubleshooting": f"I keep getting error {meta.get('code')} and nothing syncs. What does it mean and how do I fix it?",
    }[article["kind"]]


def conversation(rng: random.Random) -> dict:
    product = rng.choice(list(PRODUCTS))
    articles = knowledge_base(product)
    messages = [{"role": "system", "content": system_prompt(product)}]
    if rng.random() < 0.3:    # a prior turn, like a real multi-turn chat
        messages += [
            {"role": "user", "content": _question(rng, product, articles)},
            {"role": "assistant", "content": f"Thanks for reaching out. I checked the {product} knowledge base for you; could you confirm which workspace this is about?"},
            {"role": "user", "content": "It's our main production workspace."},
        ]
    messages.append({"role": "user", "content": _question(rng, product, articles)})
    return {"product": product, "messages": messages, "tools": TOOLS}


def target_conversations(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(f"targets:{seed}")
    return [conversation(rng) for _ in range(n)]


def filler_conversations(n: int, seed: int = 1) -> list[dict]:
    """Background traffic: same products (so prefixes are shared), mixed sampling settings and lengths."""
    rng = random.Random(f"fillers:{seed}")
    out = []
    for _ in range(n):
        conv = conversation(rng)
        if rng.random() < 0.5:
            conv["sampling"] = {"temperature": 0.0}
        else:
            conv["sampling"] = {"temperature": rng.choice([0.6, 0.8, 1.0]), "top_p": rng.choice([0.9, 0.95, 1.0]), "seed": rng.randrange(2**31)}
        conv["max_tokens"] = rng.choice([64, 128, 256, 512, 768])
        out.append(conv)
    return out


def render(tokenizer, conv: dict) -> str:
    return tokenizer.apply_chat_template(conv["messages"], tools=conv.get("tools"), add_generation_prompt=True, tokenize=False, enable_thinking=False)
