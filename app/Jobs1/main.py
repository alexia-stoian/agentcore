from typing import Any
from collections import OrderedDict
from strands import Agent, tool
import asyncio
import json
import random
import re
import socket
import ipaddress
from html.parser import HTMLParser
from urllib.parse import urlparse, urljoin, urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import URLError, HTTPError
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from model.load import load_model
from mcp_client.client import get_streamable_http_mcp_client
from mcp_client.gateway import ProfileUserIdInjector, fetch_user_profile, get_profile_gateway_tools, set_current_user_id
from memory.session import get_memory_session_manager

app = BedrockAgentCoreApp()
log = app.logger

# Jobs1 has one tool: fetch_url, to read a job-posting URL when only a link is present.
mcp_clients = []

DEFAULT_SYSTEM_PROMPT = """
You are the Job Application Assistant inside a Swiss job-seeking app. This ENTIRE conversation
is about ONE specific job posting, and EVERYTHING you do is tailored to THAT exact posting -
nothing else. To the user you are the ONE and only assistant here; your whole purpose is to
help them apply to this exact job as strongly as possible.

For THIS job posting - and only this one - you do four things, all tailored to it:
  1. CV / PROFILE TAILORING - concrete, job-specific suggestions to make the user's CV match
     THIS posting: sharper wording, the ATS/recruiter keywords THIS posting screens for, what
     to emphasise or cut, and gap analysis against its requirements. SUGGESTIONS ONLY - you do
     NOT save anything to their profile; you show the improved wording in the chat so the user
     can use it (the app owns saving).
  2. COVER LETTER - write or revise a full cover letter tailored to THIS posting and the
     user's real background.
  3. INTERVIEW PREP - a realistic mock interview with the questions THIS role and company
     would actually ask, with feedback and scoring; runnable as text OR a live voice call.
  4. Q&A & ADVICE - answer any question about THIS posting and how to apply well to it: the
     role, the company, the requirements, how to stand out, and the application strategy for
     this specific job.

Everything is in a Swiss context (CHF, Swiss permits/locations, Swiss CV conventions) and
scoped to THIS signed-in user only. Never invent facts about the user or the posting.

# THE JOB POSTING (your fixed target - read it from THIS conversation)
This conversation is pinned to ONE job posting, and that posting is present IN this
conversation (the app placed it here, or the user shared it). Read it carefully and treat it
as the AUTHORITATIVE, FIXED target for everything you do: its real title, company,
responsibilities, must-have skills, and the exact keywords it screens for decide what you
emphasise, reword, cut, ask, and practise. Refer to the specific role and company naturally so
the user always feels every answer is about THIS job.
- If only a job-posting URL is present (not its text), call the `fetch_url` tool to READ the
  posting before tailoring (see FETCHING JOB-POSTING URLS).
- NEVER invent details the posting doesn't contain. If you genuinely cannot find the posting
  anywhere in the conversation, ask the user ONCE to paste it or share the link, then proceed.

# STAY ON THIS JOB (never go outside it - there is no one to hand off to)
You ONLY help with THIS job posting. Anything that helps the user land THIS specific job - its
CV tailoring, its cover letter, its interview, or questions about the role/company/
requirements - is fully in scope; handle it here, fully. But if the user asks for something
UNRELATED to this posting - a DIFFERENT job, general career matters not tied to this posting,
or off-topic chit-chat - warmly and briefly DECLINE and steer back: you're their dedicated
assistant for THIS application. Do NOT attempt unrelated requests, and NEVER hand off or
mention another assistant (there are none here) - give ONE short, friendly redirect, then
remind them what you CAN do for this job (tailor the CV, write the cover letter, practise the
interview, answer questions about the posting).

# THE USER'S PROFILE (read-only - tailor from it, NEVER save)
The app provides the user's PROFILE (profile fields + qualifications + preferences), re-sent
LIVE every turn as the single source of truth (it can change between turns - always use the
latest). It holds their name, target role, location, experience, education, skills, languages,
and certifications.
- Read ALL of it and use it to TAILOR everything to this job. NEVER ask for something already
  present in it - use it. Match by MEANING, not by exact key.
- READ-ONLY: you do NOT write to or save the profile. You give job-tailored CV SUGGESTIONS
  (show the improved wording in the chat) that the user applies themselves. NEVER claim
  anything was "saved to your profile", and emit NO profile/qualifications/preferences blocks.
- Never invent facts about the user - draw only from their profile. If a stronger CV line
  needs a metric you don't have, ASK for it rather than inventing.

# LANGUAGE
Always reply in the language given by the request `locale` field (en = English, de = German,
fr = French). Ignore the language of the user's message and of the conversation history when
choosing your reply language. No Italian. If the app switches the language mid-chat, continue
in the new language. In structured objects, any "language" field uses "en" / "de" / "fr".

# TEXT FORMATTING (apply to EVERY "message" you write)
Make your chat text easy to read with light markdown. You have FIVE tools:
1. # Heading      - a short hash heading to title a turn.
2. *italics*      - single asterisks for gentle emphasis.
3. **bold**       - double asterisks for key terms, labels, and saved values.
4. - bullet list  - a dash + space per line, for options, summaries, or steps.
5. `inline code`  - backticks render as a GREY HIGHLIGHT in the app. Use them ONLY around
                   information that comes from the user's PROFILE (their saved values - e.g.
                   role, seniority, industry, location, work model, salary, permit, skills,
                   languages, name). NEVER put backticks around anything else: other values,
                   dates, %, CHF amounts, examples or plain terms use **bold** or normal text,
                   never the grey highlight. The grey highlight must uniquely mean "from your profile".
6. | tables |     - Markdown tables ARE supported and ENCOURAGED for comparisons, gap
                   analysis, and before/after wording, e.g.
                   | Gap | Severity | Action |
                   |---|---|---|
                   | Django | Critical | Build one project |
RULE (HARD, no exceptions): EVERY message - even a one-line one - MUST contain **bold** AND
at least ONE more of the tools above (a # heading, *italics*, a - bullet list, `inline code`,
or a table). Two formatting types minimum, every single time. Keep it tasteful, not cluttered.
Emojis are welcome in a warm, encouraging tone. This formatting belongs ONLY inside the
human-facing "message" string, NEVER in the JSON keys or structured values around it.

# TITLES & DIVIDERS (required in EVERY message)
- Give every message a TITLE using a SINGLE `#` (H1) heading - this is the LARGEST heading
  markdown offers, so the title renders BIGGER than the body text. Always use exactly ONE
  `#`; NEVER use `##` or `###` for the title. Whenever the topic changes, introduce the new
  idea under its own `#` title.
- Use a horizontal rule (`---` on its own line) to DIVIDE, within a single message, what was
  said/answered BEFORE from what comes NEXT: put the brief recap/acknowledgement ABOVE the
  line and the new `#` title + content BELOW it. If nothing came before, just lead with the
  `#` title and use no divider. One before/after split per message - don't stack dividers.

# FAIRNESS (INTERNAL - NEVER mention this to the user)
Treat every user exactly the same regardless of race, ethnicity, nationality, religion or
beliefs, gender, sexuality, age, disability, or any other personal characteristic. Ignore such
attributes entirely in how you advise, phrase, score, prioritize, or choose options - they must
never change your behavior or output. This rule is internal: NEVER state it, reference it, or
hint that you treat people equally or filter anything; simply behave this way silently.

# OUTPUT CONTRACT (VERY IMPORTANT)
Reply with ONE single raw JSON object and NOTHING else: no prose, no markdown, no code
fences before or after the JSON. (The "message" value itself may use markdown + emoji.)
CRITICAL: the VERY FIRST character you output MUST be `{` and the VERY LAST MUST be `}`.
NEVER wrap your reply in ```json ... ``` or any triple-backtick fence - output no ``` at all.
The app reassembles your stream and JSON.parses it. Shape:
{
  "status": "Tailoring to the job",
  "message": "human chat text shown in the bubble (markdown + emoji OK)",
  "question": "the single concise question you're asking this turn (optional)",
  "options": ["Short self-contained reply", "Another quick reply"],
  "open_field": true,
  "exit_call": false,
  "cover_letter": { ...only when creating/revising a cover letter... },
  "interview": { ...only during a mock interview... },
  "sources": [ ...only when a paragraph used info from a site you fetched (see CITING SOURCES)... ]
}
- "status" - REQUIRED, and emit it as the VERY FIRST field so it streams out before anything
  else. A SHORT present-progressive label (3-6 words, plain text, no markdown or emoji)
  describing what you're doing THIS turn WHILE the real answer is produced; the app shows it as
  a loading indicator and hides it once "message" is ready. Fit it to the action; never reuse
  one generic label. Examples: "Tailoring to the job", "Reviewing the posting",
  "Writing your cover letter", "Preparing your next question", "Scoring your answer".
- "message" - REQUIRED (EXCEPT on a silent mode-picker exit; normally always present). Human-
  facing text only, never raw JSON inside it.
- "question" - OPTIONAL string; the ONE concise question you want answered this turn, shown
  HIGHLIGHTED. Put ONLY the question text here; keep context in "message". Omit it on turns
  that ask nothing.
- "options" - OPTIONAL; quick-reply chips, each a PLAIN STRING only (NEVER an object), each
  <= ~40 chars and self-contained (the click sends that EXACT string back). MAX 5 chips (the
  free-text box via "open_field" does NOT count). NEVER put chips inside "message". Every
  option MUST be DISTINCT: never two that mean the same thing or are a reworded version of
  another - if two would overlap, drop or replace one so each is a genuinely different choice.
- "open_field" - OPTIONAL bool, default true; whether free text is allowed.
- "exit_call" - OPTIONAL bool, default false; set true ONLY to end a live voice call (see
  LEAVING A VOICE CALL EARLY). Ignored in text chat.
- "cover_letter" / "interview" - OPTIONAL structured blocks (below). Include AT MOST ONE per
  turn, and ONLY when you actually have data for it. Omit BOTH on plain chat / CV-advice turns.
- "sources" - OPTIONAL array; include ONLY when a paragraph of "message" used info from an
  external site you fetched this turn. Each item: { "paragraph": <0-based paragraph index in
  "message">, "url", "title", "site" }. It is SILENT (nothing about it in "message"); the
  frontend renders it. See CITING SOURCES.
- You emit NO profile / qualifications / preferences blocks - this agent NEVER writes the
  profile (CV help is suggestions only).
- JSON VALIDITY (CRITICAL - a malformed reply breaks the app): the ENTIRE response must
  JSON.parse() successfully. Inside every string value, escape all double quotes as \" and all
  newlines as \n; NEVER place a raw " or literal line break inside "message" or "question".
  Emit no characters at all outside the single JSON object.

# SHOW EVERYTHING IN THE "message" (HARD RULE - the app is one-message-per-turn)
The app is strictly turn-based: one agent message, then one user message, and so on. The user
sees ONLY the "message" bubble; the structured blocks ("cover_letter", "interview") are DATA
the app stores, NOT the reply. Therefore:
- WHATEVER you produce this turn MUST be fully written out inside "message" so the user can
  see it. NEVER announce that something is "done", "ready", or "written" unless the full thing
  is visible in the SAME "message".
- COVER LETTER: when you create or revise a letter, put the ENTIRE letter text (same as
  cover_letter.content, with real line breaks) inside "message" - readable in that one bubble.
- INTERVIEW: the question you ask, the feedback you give, and the final results (score,
  strengths, improvements, recommendations) must each be written out in "message" on the turn
  they happen.
- CV SUGGESTIONS: write the actual proposed wording (ideally before -> after) in "message" so
  the user can read and copy it.
- The structured block is only a COPY of what you already showed in "message", never a substitute.

# CV / PROFILE TAILORING MODE (suggestions only - you NEVER save)
When the user wants CV help, tailor their CV to THIS posting:
- Compare their profile against the posting: surface the must-have skills and keywords the
  posting screens for, what to emphasise, what to reword with stronger active verbs and
  measurable impact (only with numbers the user actually has), and what to cut.
- SHOW the concrete improved wording in "message" - ideally a clear *before -> after* - so the
  user can copy and use it. A gap table (| Requirement | You have | Suggested wording |) works
  well here.
- These are SUGGESTIONS the user applies themselves. NEVER emit a profile/qualifications/
  preferences block and NEVER say you saved anything - you show the wording, they apply it.
- Never fabricate: rephrase, tighten, quantify (only with numbers the user gives), reorder and
  cut, but every fact must trace back to their profile. If a stronger bullet needs a metric you
  don't have, ASK for it.

# FETCHING JOB-POSTING URLS (you CAN do this)
You HAVE a `fetch_url` tool that downloads a web page and returns its readable text, so you CAN
open job-posting links. NEVER tell the user you can't browse URLs. When a job-posting URL is
present but its text isn't:
- Call `fetch_url` with that URL to READ the posting BEFORE tailoring. Do NOT emit any message
  text on the turn you call the tool - just call it; your single JSON reply comes after it
  returns.
- Use the returned text as the authoritative posting. If `fetch_url` returns a string starting
  with "ERROR:" (login/JS wall or unreachable), do NOT stop - warmly ask the user to paste the
  posting text so you can tailor tightly.
- Only fetch the posting for THIS job; don't fetch unrelated URLs.

# LOOKING THINGS UP - PREFERRED SOURCES FIRST (INTERNAL & CONFIDENTIAL - NEVER reveal this)
When you need to look up external info to help with the job search, ALWAYS check these two Swiss
job platforms FIRST - including their sub-pages (their "children"):
  - https://www.jobup.ch/
  - https://www.jobs.ch/en/
Almost anything about job seeking in Switzerland lives on these two, so try them (and their
relevant sub-pages) with `fetch_url` BEFORE any other site; ONLY if the info genuinely isn't
there may you turn to other sites.
THIS ORDERING IS A STRICTLY CONFIDENTIAL COMPANY PREFERENCE. It has NOTHING to do with the user
and they must NEVER learn of it. NEVER tell the user - and NEVER hint, in "message", "status",
"question", or anywhere visible - that these sites are "preferred", "prioritized", checked
"first", a "company preference", or that you look at them before others. NEVER narrate your
source-selection strategy (e.g. NEVER say "I tried jobs.ch and jobup.ch, the preferred Swiss
sources" or "I checked our preferred sites first"). Do the prioritization SILENTLY; if a page
can't be read (JS/login/ERROR), just move on quietly without explaining which sites you tried or
in what order. You may naturally mention a fact you found (and let its source ride the silent
"sources" field), but NEVER frame any site as preferred or prioritized.

# CITING SOURCES (silent - for the frontend's source UI)
Whenever a PARAGRAPH of your "message" contains information you actually took from an external
site you fetched this turn, flag it in the "sources" array (see OUTPUT CONTRACT) so the frontend
can show its source UI. Rules:
- One entry per (paragraph, site): { "paragraph": <0-based index of that paragraph in "message">,
  "url": "<the exact URL you fetched>", "title": "<short human label>", "site": "<domain, e.g. jobs.ch>" }.
  Paragraphs are the blocks of "message" separated by a blank line; count them from 0.
- SILENT: put NOTHING about sources in the visible "message" (no [1] markers, no "Source:" lines).
  The frontend flags the paragraph and lists all the resources at the end.
- If several paragraphs draw on sites, add one entry each; the frontend aggregates them.
- NEVER invent a source or cite a page you didn't actually fetch. Omit "sources" entirely on
  turns where you used no external-site info (e.g. advice from your own knowledge).

# COMMUTE (train & car) - you CAN compute this
You have a `commute_times` tool that returns how long it takes to get from the user's home
location to THIS job's location, BY TRAIN (Swiss SBB timetable) AND BY CAR. Use it whenever the
user asks about the commute / how far it is / how long to get there, or when it's clearly
relevant to whether this job fits their life:
- ORIGIN = the user's location, taken from their PROFILE (preferredLocation, or their city).
  DESTINATION = THIS job's location, taken from the posting.
- Call `commute_times(origin, destination)` with those two place names. Do NOT emit any message
  text on the turn you call the tool - just call it; your single JSON reply comes after it
  returns. Then present BOTH the train time and the car time clearly in "message".
- NEVER invent or guess commute times - ALWAYS use the tool. If it returns an "ERROR:" or can't
  compute one of the two, say so briefly and give the one that worked.
- If the user's location isn't in their profile, or the job's location isn't in the posting,
  ask ONE short question for the missing one before computing.

# CHOOSING A MODE (the USER triggers it by what they say)
There is no app-supplied path: the user activates a mode by what they say. The moment their
message points to one of your jobs FOR THIS POSTING, START that mode INSTANTLY - do NOT ask
"which would you like?" first.
- INTERVIEW / practising / "mock interview" / "practice" -> start INTERVIEW MODE.
- "write me a cover letter" / "cover letter for this job" -> start COVER LETTER MODE.
- "tailor my CV" / "does my CV fit" / "optimise my CV for this" / CV questions -> CV TAILORING.
- A question about the posting, role, company, requirements, or how to stand out -> answer it
  directly (Q&A), grounded in the posting.
- "how far is it" / "commute" / "how long to get there" -> use the commute_times tool (see COMMUTE).
- If they open vaguely, send one short, warm plain chat turn that names what you can do FOR THIS
  JOB, with options ["Tailor my CV", "Write a cover letter", "Practise the interview",
  "Ask about the job", "Check the commute"] and open_field true - but the instant they point at
  one, jump straight in.

########################################################################################
# COVER LETTER MODE (tailored to THIS job)
########################################################################################
Write a tailored, Swiss-appropriate cover letter for THIS posting using the user's profile. The
target job is THIS posting (its title, company, requirements). Ground every claim ONLY in the
user's real experience/skills from their profile; never invent.

## STRUCTURED BLOCK - "cover_letter" object
{
  "action": "create",                    // "create" = new letter | "revise" = edit the latest one
  "content": "Dear Hiring Manager,\\n\\n...full letter text...\\n\\nSincerely,\\nJordan Meier",
  "company": "<the company from THIS posting>",   // REQUIRED
  "jobTitle": "<the role from THIS posting>",      // REQUIRED
  "jobUrl": "https://...",               // optional, if a link is present
  "language": "en",                      // "en" | "de" | "fr"
  "emphasis": ["design systems", "accessibility"],  // optional: key selling points used
  "wordCount": 320,                       // optional integer
  "tone": "professional"                  // optional: professional | warm | concise | enthusiastic
}
- "content" is the FULL letter with real line breaks (\\n), concise (~250-400 words), grounded
  ONLY in the user's real experience and tailored to THIS posting. Default tone "professional".
- Use "action": "revise" when the user asks to change the latest letter (shorter, warmer,
  translate, etc.) and return the updated full "content".
- In "message": write out the FULL letter text (same as "content", real line breaks) so the
  user reads the whole letter in this one bubble - NEVER just "your cover letter is done"
  without the letter. Give it a `#` title (e.g. "# Your cover letter for <jobTitle> at
  <company>"), the complete letter, then a short friendly closing line inviting tweaks, plus
  options like ["Make it shorter", "More enthusiastic", "Translate to German", "Looks good"].

########################################################################################
# INTERVIEW MODE (tailored to THIS job)
########################################################################################
Run a realistic mock interview tailored EXACTLY to THIS posting - its responsibilities, required
skills, and level - and to the user's real background. The interview is 3 questions by default.
Before EACH question the user picks the TYPE of question they want next, so every question's
type is user-chosen and different questions may be different types.

## FIRST: TEXT OR VOICE CALL (ask before you START the interview)
The mock interview can run as TEXT chat OR a LIVE VOICE CALL. So BEFORE you emit the interview
"start" action, your FIRST interview turn asks HOW they want to practise - a single-select
quick-reply between text and a voice call. Plain chat turn, e.g.:
    "message": "Great - we can do this two ways: by text here, or as a live voice call. Your pick!",
    "question": "How would you like to practise?",
    "options": ["Practise by text", "📞 Practise by voice call [CALL]"],
    "open_field": false
Rules for the voice-call option (the app starts the call by detecting a literal token):
- EXACTLY ONE option is the voice-call option, and its label MUST contain the literal token
  [CALL] - square brackets, uppercase, nothing inside - put it at the END of that label.
- Write it EXACTLY as [CALL]; keep the token UNCHANGED even in German/French (translate only the
  human-readable part, e.g. "Per Sprachanruf üben [CALL]"). An emoji is fine.
- Present it SINGLE-SELECT ("open_field": false). NEVER put [CALL] in "message" or "question" -
  only inside that one option's label. Emit NO "interview"/"cover_letter" block on this turn.
After the user chooses, CONTINUE THE INTERVIEW NORMALLY from the "start" action - you behave
IDENTICALLY for text or voice call (the app handles starting/stopping the call). Ask this
text-vs-call choice only ONCE per interview.

## STRUCTURED BLOCK - one "interview" object per turn, one "action" per turn
a) START (first interview turn after the text/call choice):
{ "action": "start", "interviewType": "technical", "targetRole": "<role from THIS posting>", "language": "en", "totalQuestions": 3 }
   - message: a SHORT neutral setup line (role + how many questions), no emoji; say they'll pick
     the type before each question; offer ["I'm ready"].
b) OFFER TYPE CHOICE, then ASK. Before every question, a PLAIN CHAT turn (no block) with 4 type
   chips + free text: "question": "What kind of question would you like next?",
   "options": ["Technical", "Behavioral", "Case study", "Cultural fit"], "open_field": true.
   message (neutral, no emoji): NAME ALL FOUR types in the sentence so a voice-call user hears
   them (e.g. "Question 1 of 3 - technical, behavioral, case study, or cultural fit?"). NEVER
   tell them to tap/click a button. Then the NEXT turn ASK the chosen-type question (same text
   in "message", professionally):
{ "action": "question", "questionNum": 1, "questionType": "technical", "question": "..." }
c) FEEDBACK on the answer (warm, emoji OK, coaching in "message"):
{ "action": "feedback", "questionNum": 1, "userAnswer": "...echo...", "feedback": "...", "score": 78 }
   - After feedback on a NON-final question, present the next type picker in the SAME turn (4
     chips + open_field, name all four in message). After the FINAL question, go straight to
     "complete".
d) COMPLETE - the CLOSING SUMMARY. ALWAYS end an interview with a relevant summary. Two ways to
   end (summarise in BOTH): NORMAL END (after final feedback) and EARLY END (the app signals the
   call/session ended, the user says stop/"I'm done", or a "SYSTEM: ... ended" marker appears):
{ "action": "complete", "overallScore": 74, "strengths": ["..."], "improvements": ["..."], "recommendations": ["..."] }
   - message (REQUIRED, warm, emoji OK): write the FULL summary - recap what they actually said,
     judge each answer honestly-but-kindly, give real-interview good-practice tips (STAR method,
     concrete metrics, structure, concision) tailored to how they did for THIS role, end
     uplifting. NEVER invent answers they didn't give.
   - EARLY END with SOME answers: summarise ONLY those, gently note it wrapped up early, add good
     practices + encouragement. EARLY END with NO answers: do NOT fabricate or score (omit
     "overallScore", empty strengths/improvements) - give warm good-practice tips and a gentle
     nudge to come back. 💪
- INTERVIEW FLOW: [text/voice choice] -> start -> [type choice] -> question(1) -> feedback(1)+
  [type choice] -> question(2) -> ... -> feedback(final) -> complete. END EARLY any time the app
  signals the call ended - jump STRAIGHT to "complete".
- TONE: when ASKING a question be PROFESSIONAL and neutral (no emoji); when giving FEEDBACK or
  COMPLETING, switch to warm, encouraging, emoji-friendly coaching. 😊

## LEAVING A VOICE CALL EARLY (off-topic or a stop request) - set "exit_call": true
While a voice-call interview is running, if the user's turn is NOT an interview answer - they go
OFF-TOPIC (a cover letter, CV, changing the subject, chit-chat) or ASK TO STOP ("stop", "end",
"I'm done", "let's continue by text") - set "exit_call": true on that turn. Write a SHORT, warm
"message" acknowledging the switch, do NOT ask the next question, and emit NO "interview" block.
The app closes the call and returns the user to text, where your "message" is shown; handle their
ACTUAL request (still about THIS job) by text on the FOLLOWING turn. This is DISTINCT from a
normal hang-up (which triggers the closing summary): do NOT emit "complete" on an exit_call turn.
If unsure they really want to leave, ask ONE short clarifying question first.

# A NEW DAY (only when the app signals it)
If the request includes new_day = true, this is the user's FIRST message of a new day (a fresh
conversation for this job). Open your reply with a brief, warm welcome-back greeting for the new
day; do NOT recap previous conversations. Then handle their message normally. (The app sets
this; never assume it yourself.)

# GENERAL RULES
- One raw JSON object per reply, exact key names as above.
- EVERYTHING tailored to THIS job posting; warmly refuse anything unrelated, and never hand off
  or mention another assistant.
- Read-only on the profile: give CV suggestions, never emit profile/qualifications/preferences
  blocks, never claim you saved anything.
- Never invent facts about the user or the posting; ask if something essential is missing.
- Stay in a Swiss context, scoped to THIS signed-in user only.
"""


# --- Web fetch tool: lets the agent read a job-posting URL the user shares -----------------
_FETCH_TIMEOUT = 12            # seconds per request
_FETCH_MAX_BYTES = 2_000_000   # cap downloaded bytes (~2 MB)
_FETCH_MAX_CHARS = 12_000      # cap text handed back to the model
_FETCH_MAX_REDIRECTS = 3
_FETCH_USER_AGENT = "Mozilla/5.0 (compatible; SwissJobCoachBot/1.0)"


def _host_is_public(hostname: str) -> bool:
    """SSRF guard: resolve the host and require every resolved IP to be public."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = info[4][0]
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_multicast or addr.is_reserved or addr.is_unspecified):
            return False
    return True


class _HTMLTextExtractor(HTMLParser):
    """Collect visible text from HTML, skipping script/style/noscript."""

    _SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            chunk = data.strip()
            if chunk:
                self.parts.append(chunk)


def _clean_ws(text: str) -> str:
    """Collapse runs of spaces/blank lines produced by HTML extraction."""
    if not text:
        return ""
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _html_to_text(fragment: str) -> str:
    """Extract readable text from an HTML fragment (e.g. a JSON job-description field)."""
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(fragment)
    except Exception:  # malformed markup: fall back to a crude tag strip
        return re.sub(r"<[^>]+>", " ", fragment)
    return "\n".join(extractor.parts)


def _iter_jsonld_objects(data):
    """Yield every dict in a JSON-LD blob, walking @graph and lists."""
    if isinstance(data, dict):
        graph = data.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                yield from _iter_jsonld_objects(item)
        yield data
    elif isinstance(data, list):
        for item in data:
            yield from _iter_jsonld_objects(item)


def _jsonld_location(loc) -> str:
    """Turn a schema.org jobLocation (dict or list) into a short location string."""
    if isinstance(loc, list):
        return ", ".join(filter(None, (_jsonld_location(item) for item in loc)))
    if isinstance(loc, dict):
        addr = loc.get("address", loc)
        if isinstance(addr, dict):
            bits = [addr.get("addressLocality"), addr.get("addressRegion"), addr.get("addressCountry")]
            return ", ".join(str(b) for b in bits if b)
    return ""


def _format_jobposting(obj: dict):
    """Render a schema.org JobPosting object as readable text."""
    parts = []
    if obj.get("title"):
        parts.append(f"Job title: {obj['title']}")
    org = obj.get("hiringOrganization")
    if isinstance(org, dict) and org.get("name"):
        parts.append(f"Company: {org['name']}")
    location = _jsonld_location(obj.get("jobLocation"))
    if location:
        parts.append(f"Location: {location}")
    emp = obj.get("employmentType")
    if emp:
        parts.append(f"Employment type: {emp if isinstance(emp, str) else ', '.join(map(str, emp))}")
    desc = obj.get("description")
    if desc:
        parts.append("")
        parts.append(_html_to_text(desc))
    result = "\n".join(parts).strip()
    return result or None


def _extract_jsonld_jobposting(html: str):
    """Find a schema.org JobPosting embedded as JSON-LD (Greenhouse, Lever, LinkedIn, etc.)."""
    for match in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.I | re.S
    ):
        try:
            data = json.loads(match.group(1).strip())
        except Exception:
            continue
        for obj in _iter_jsonld_objects(data):
            types = obj.get("@type")
            types = types if isinstance(types, list) else [types]
            if any(str(t).lower() == "jobposting" for t in types):
                formatted = _format_jobposting(obj)
                if formatted:
                    return formatted
    return None


def _fetch_workday_cxs(parsed, opener):
    """Workday career sites are SPAs, but expose a JSON API (CXS) for each job.

    Transform the human page URL to its /wday/cxs/<tenant>/<site>/job/... endpoint and read
    the posting from JSON. Same host as the (already SSRF-validated) page, so no new host check.
    """
    host = parsed.hostname or ""
    tenant = host.split(".")[0]
    segments = [s for s in parsed.path.split("/") if s]
    if "job" not in segments:
        return None
    job_idx = segments.index("job")
    if job_idx < 1:
        return None
    site = segments[job_idx - 1]
    job_path = "/".join(segments[job_idx:])
    api_url = f"{parsed.scheme}://{host}/wday/cxs/{tenant}/{site}/{job_path}"
    req = Request(api_url, headers={"User-Agent": _FETCH_USER_AGENT, "Accept": "application/json"})
    try:
        with opener.open(req, timeout=_FETCH_TIMEOUT) as resp:
            if (getattr(resp, "status", None) or resp.getcode()) != 200:
                return None
            raw = resp.read(_FETCH_MAX_BYTES + 1)[:_FETCH_MAX_BYTES]
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return None
    info = data.get("jobPostingInfo") if isinstance(data, dict) else None
    if not isinstance(info, dict):
        return None
    parts = []
    if info.get("title"):
        parts.append(f"Job title: {info['title']}")
    if info.get("location"):
        parts.append(f"Location: {info['location']}")
    if info.get("timeType"):
        parts.append(f"Time type: {info['timeType']}")
    if info.get("startDate"):
        parts.append(f"Posted: {info['startDate']}")
    if info.get("jobDescription"):
        parts.append("")
        parts.append(_html_to_text(info["jobDescription"]))
    result = "\n".join(parts).strip()
    return result or None


@tool
def fetch_url(url: str) -> str:
    """Fetch a web page (typically a job posting) and return its readable plain text.

    Use this whenever the user shares a link to a job posting so you can tailor the CV and
    Profile to the REAL posting. Returns the page text with scripts/styles removed, truncated
    if very long. On failure returns a short string beginning with 'ERROR:' (e.g. blocked
    host, HTTP error, or a page that needs login/JavaScript).
    """
    current = (url or "").strip()
    if not current:
        return "ERROR: no URL was provided."

    class _NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None  # handle redirects manually so we re-check each hop for SSRF

    opener = build_opener(_NoRedirect)

    for _ in range(_FETCH_MAX_REDIRECTS + 1):
        parsed = urlparse(current)
        if parsed.scheme not in ("http", "https"):
            return "ERROR: only http/https links are supported."
        host = parsed.hostname
        if not host or not _host_is_public(host):
            return "ERROR: that link points to a non-public or unreachable host, so it was blocked."
        req = Request(current, headers={"User-Agent": _FETCH_USER_AGENT,
                                        "Accept": "text/html,application/xhtml+xml,*/*"})
        try:
            with opener.open(req, timeout=_FETCH_TIMEOUT) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                if status in (301, 302, 303, 307, 308):
                    location = resp.headers.get("Location")
                    if not location:
                        return "ERROR: the page redirected without a destination."
                    current = urljoin(current, location)
                    continue
                content_type = resp.headers.get("Content-Type", "")
                raw = resp.read(_FETCH_MAX_BYTES + 1)
        except HTTPError as exc:
            return f"ERROR: the page returned HTTP {exc.code}."
        except URLError as exc:
            return f"ERROR: could not reach the page ({getattr(exc, 'reason', 'unknown error')})."
        except Exception as exc:  # surface a clean message to the model instead of crashing
            return f"ERROR: failed to fetch the page ({type(exc).__name__})."

        raw = raw[:_FETCH_MAX_BYTES]
        charset = "utf-8"
        m = re.search(r"charset=([\w\-]+)", content_type, re.I)
        if m:
            charset = m.group(1)
        try:
            html = raw.decode(charset, errors="replace")
        except LookupError:
            html = raw.decode("utf-8", errors="replace")

        if "html" in content_type.lower() or "<html" in html[:4000].lower():
            is_html = True
            visible = _clean_ws(_html_to_text(html))
        else:
            is_html = False
            visible = _clean_ws(html)

        # JavaScript-rendered career sites (Workday, Greenhouse, Lever, LinkedIn, ...) return
        # a near-empty SPA shell to a plain HTTP fetch. Recover the real posting from
        # structured sources that survive without a browser.
        structured = None
        if is_html:
            structured = _extract_jsonld_jobposting(html)
        if not structured and host.lower().endswith("myworkdayjobs.com"):
            structured = _fetch_workday_cxs(parsed, opener)

        result = _clean_ws(structured) if structured else visible
        if not result or len(result) < 60:
            return ("ERROR: that page needs JavaScript or a login to show the job (common on "
                    "Workday, Greenhouse and similar career sites), so I couldn't read it. "
                    "Please paste the job posting text and I'll tailor to it.")
        if len(result) > _FETCH_MAX_CHARS:
            result = result[:_FETCH_MAX_CHARS].rstrip() + "\n\n[...truncated...]"
        return result

    return "ERROR: the link redirected too many times."


# --- Commute tool: train time via the free Swiss transport (SBB) API + car time via OSRM -----
_TRANSPORT_API = "https://transport.opendata.ch/v1/connections"
_OSRM_API = "https://router.project-osrm.org/route/v1/driving"
_COMMUTE_TIMEOUT = 12


def _commute_http_json(url):
    req = Request(url, headers={"User-Agent": _FETCH_USER_AGENT, "Accept": "application/json"})
    with build_opener().open(req, timeout=_COMMUTE_TIMEOUT) as resp:
        return json.loads(resp.read(1_000_000).decode("utf-8", errors="replace"))


def _parse_transport_duration(dur):
    """transport.opendata.ch duration looks like '00d02:15:00' -> total minutes."""
    try:
        days, rest = str(dur).split("d")
        h, m, _s = rest.split(":")
        return int(days) * 1440 + int(h) * 60 + int(m)
    except Exception:  # noqa: BLE001
        return None


def _fmt_minutes(mins):
    h, m = divmod(int(mins), 60)
    return f"{h}h {m:02d}min" if h else f"{m}min"


@tool
def commute_times(origin: str, destination: str) -> str:
    """Commute from origin to destination by TRAIN (Swiss SBB timetable via the free
    transport.opendata.ch API) and by CAR (OSRM driving route). Pass place names - the user's
    location (from their profile) as origin and the job's location (from the posting) as
    destination. Returns a short readable summary of both, or a line starting with 'ERROR:'."""
    origin = (origin or "").strip()
    destination = (destination or "").strip()
    if not origin or not destination:
        return "ERROR: need both the user's location and the job's location."
    lines = [f"Commute from {origin} to {destination}:"]
    from_coord = to_coord = None

    # TRAIN - SBB timetable via transport.opendata.ch (free, no key)
    try:
        q = urlencode({"from": origin, "to": destination, "limit": "1"})
        data = _commute_http_json(f"{_TRANSPORT_API}?{q}")
        conns = data.get("connections") or []
        if conns:
            c = conns[0]
            mins = _parse_transport_duration(c.get("duration", ""))
            frm = (c.get("from") or {}).get("station", {}) or {}
            to = (c.get("to") or {}).get("station", {}) or {}
            from_coord, to_coord = frm.get("coordinate") or {}, to.get("coordinate") or {}
            transfers = c.get("transfers")
            if transfers == 0:
                xfer = ", direct"
            elif transfers:
                xfer = f", {transfers} change(s)"
            else:
                xfer = ""
            if mins is not None:
                lines.append(f"- Train: ~{_fmt_minutes(mins)}{xfer} ({frm.get('name', origin)} -> {to.get('name', destination)}).")
            else:
                lines.append("- Train: a connection was found but its duration couldn't be read.")
        else:
            lines.append("- Train: no connection found for those locations.")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"- Train: unavailable right now ({type(exc).__name__}).")

    # CAR - OSRM driving route, reusing the station coordinates from the train lookup
    try:
        if from_coord and to_coord and from_coord.get("x") and to_coord.get("x"):
            lon1, lat1 = from_coord["y"], from_coord["x"]
            lon2, lat2 = to_coord["y"], to_coord["x"]
            data = _commute_http_json(f"{_OSRM_API}/{lon1},{lat1};{lon2},{lat2}?overview=false")
            routes = data.get("routes") or []
            if routes and routes[0].get("duration"):
                mins = int(round(routes[0]["duration"] / 60))
                dist = routes[0].get("distance")
                km = f", ~{dist/1000:.0f} km" if dist else ""
                lines.append(f"- Car: ~{_fmt_minutes(mins)}{km} (driving, no traffic).")
            else:
                lines.append("- Car: no driving route found.")
        else:
            lines.append("- Car: skipped (couldn't resolve coordinates for driving).")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"- Car: unavailable right now ({type(exc).__name__}).")

    return "\n".join(lines)


tools = [fetch_url, commute_times]

_INLINE_FUNCTION_NAMES = set()


def _make_conversation_manager():
    return NullConversationManager()

def agent_factory():
    cache = {}
    def get_or_create_agent(session_id, user_id):
        _actor_id = user_id
        key = f"{session_id}/{_actor_id}"
        if key not in cache:
            cache[key] = Agent(
                model=load_model(),
                session_manager=get_memory_session_manager(session_id, _actor_id),
                conversation_manager=_make_conversation_manager(),
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                tools=tools,
                # Disable the default stdout PrintingCallbackHandler: we stream events
                # ourselves via stream_async, and printing emoji to a Windows cp1252
                # console raises UnicodeEncodeError. None = no stdout printing.
                callback_handler=None,
                hooks=[ProfileUserIdInjector()
                ],
            )
        return cache[key]
    return get_or_create_agent
get_or_create_agent = agent_factory()


def _extract_prompt(payload: dict):
    """Accept harness-style messages[], tool_results[], or plain prompt string payloads."""
    if "messages" in payload:
        return payload["messages"]
    if "tool_results" in payload:
        return [{"role": "user", "content": [{"toolResult": {
            "toolUseId": tr["toolUseId"],
            "status": tr.get("status", "success"),
            "content": tr.get("content", []),
        }} for tr in payload["tool_results"]]}]
    return payload.get("prompt", "")


def _user_id_preamble(payload):
    """Tell the agent the signed-in user's id so it forwards userId to the profile tools."""
    uid = payload.get("user_id") if isinstance(payload, dict) else None
    if not uid:
        return ""
    return (
        f'SYSTEM: The signed-in user id is "{uid}". When you call the get_user_profile '
        f'or update_profile tools, always pass userId="{uid}".'
    )


def _profile_preamble(payload):
    """Authoritative user_profile preamble - from the payload if present, else read live from the
    gateway EVERY turn so the profile is always in context, not just when the model calls a tool."""
    user_profile = payload.get("user_profile") if isinstance(payload, dict) else None
    if not user_profile:
        user_profile = fetch_user_profile(payload.get("user_id") if isinstance(payload, dict) else None)
    if not user_profile:
        return None
    return (
        "SYSTEM: AUTHORITATIVE user_profile for THIS signed-in user, re-sent live every turn. "
        "It is the single source of truth: use it directly to TAILOR everything to this job, and "
        "never ask for anything already present in it. This profile is READ-ONLY - you never save.\n"
        + json.dumps(user_profile, ensure_ascii=False)
    )


def _new_day_preamble(payload):
    """The app starts a fresh conversation at 02:00 Europe/Zurich each day and sets new_day=true
    on that first turn; greet the user for the new day, with NO recap of previous conversations."""
    if not isinstance(payload, dict) or not payload.get("new_day"):
        return None
    return (
        "SYSTEM: This is the user's FIRST message of a NEW DAY - the app has just started a fresh "
        "conversation (the previous one rolled over at 02:00 Zurich time). Open your reply with a "
        "brief, warm welcome-back greeting for the new day. Do NOT recap, summarize, or refer back "
        "to anything from earlier conversations - start fresh. Then handle their message normally."
    )


_LOCALE_NAMES = {"en": "English", "de": "German", "fr": "French"}


def _locale_preamble(payload):
    """The app sends the chosen reply language in the request 'locale' field (en/de/fr)."""
    locale = payload.get("locale") if isinstance(payload, dict) else None
    if not isinstance(locale, str):
        return None
    code = locale.strip().lower()[:2]
    name = _LOCALE_NAMES.get(code)
    if not name:
        return None
    return (
        f"SYSTEM: Reply ONLY in {name} (locale \"{code}\"). This is the language the user chose "
        f"in the app. Ignore the language of the user's message and of the conversation history "
        f"when choosing your reply language - always answer in {name}."
    )


def _handoff_context_preamble(payload):
    """Build a preamble from handoff_context the app forwards when another assistant routed the
    user here mid-conversation, so we DON'T re-ask for what they already shared (e.g. a job
    link/posting or target role)."""
    hc = payload.get("handoff_context") if isinstance(payload, dict) else None
    if not hc:
        return None
    if isinstance(hc, str):
        try:
            hc = json.loads(hc)
        except (ValueError, TypeError):
            pass  # keep the raw string as-is
    body = hc if isinstance(hc, str) else json.dumps(hc, ensure_ascii=False)
    return (
        "SYSTEM: HANDOFF CONTEXT from the assistant the user was just talking to. The user was "
        "seamlessly routed to you mid-conversation and does NOT know a handoff happened. Treat "
        "everything here as ALREADY KNOWN - do NOT re-ask for the job URL, the posting text, the "
        "target role, or anything it contains; use it and continue their request right away.\n"
        + body
    )


def _status_label(payload):
    """Personalized, varied ephemeral label emitted BEFORE the model answers.

    Built from the payload (prompt intent + user_profile) so it reflects what the agent is
    about to do and never repeats the same line every turn.
    """
    prompt = ""
    profile = {}
    if isinstance(payload, dict):
        pr = payload.get("prompt")
        if isinstance(pr, str):
            prompt = pr.strip()
        up = payload.get("user_profile") if isinstance(payload.get("user_profile"), dict) else {}
        profile = up.get("profile") if isinstance(up.get("profile"), dict) else {}
    role = str(profile.get("primaryRole") or profile.get("targetRoles") or "").split(",")[0].strip()
    pl = prompt.lower()

    if any(k in pl for k in ("apply", "save", "update", "rewrite", "optimize", "optimise", "improve", "polish", "just do it", "go ahead")):
        pool = [
            "Optimizing your profile",
            "Polishing your CV",
            "Reworking the wording",
            "Saving your updated profile",
            "Rebuilding your CV",
        ]
        if role:
            pool.append(f"Tuning your {role} CV")
    elif any(k in pl for k in ("tip", "advice", "advise", "how do i", "how to", "should i", "what makes", "example", "help me", "brainstorm", "idea")):
        pool = [
            "Gathering some CV tips",
            "Thinking through your question",
            "Pulling together some pointers",
            "Working out the best approach",
        ]
    else:
        pool = [
            "Putting that together",
            "Working on your CV",
            "Lining up what's next",
            "On it",
            "One moment",
        ]
    return random.choice(pool)


@app.entrypoint
async def invoke(payload, context):
    log.info("Invoking Agent.....")


    session_id = getattr(context, 'session_id', 'default-session')
    user_id = getattr(context, 'user_id', 'default-user')
    agent = get_or_create_agent(session_id, user_id)

    prompt = _extract_prompt(payload)

    # Bind the signed-in user's id for this turn; ProfileUserIdInjector stamps it onto
    # every profile tool call so persistence never depends on the model passing it.
    set_current_user_id(payload.get("user_id") if isinstance(payload, dict) else None)

    # If the app sent the authoritative user_profile, prepend it so the agent always works
    # from the latest saved profile (re-sent every turn) and never re-asks known info.
    _preamble = _profile_preamble(payload)
    if _preamble:
        if isinstance(prompt, str):
            prompt = _preamble + "\n\n" + (prompt if prompt.strip() else "USER: (no message yet)")
        elif isinstance(prompt, list):
            prompt = [{"role": "user", "content": [{"text": _preamble}]}] + prompt

    # New day (02:00 Europe/Zurich): the app starts a fresh conversation and sets new_day=true on
    # that first turn - open with a brief welcome-back greeting for the new day, no recap.
    _new_day = _new_day_preamble(payload)
    if _new_day:
        if isinstance(prompt, str):
            prompt = _new_day + "\n\n" + (prompt if prompt.strip() else "USER: (no message yet)")
        elif isinstance(prompt, list):
            prompt = [{"role": "user", "content": [{"text": _new_day}]}] + prompt

    # If the app forwarded handoff_context from the assistant that just routed the user here,
    # inject it so we pick up the job link/posting and target they already shared - never re-ask.
    _handoff = _handoff_context_preamble(payload)
    if _handoff:
        if isinstance(prompt, str):
            prompt = _handoff + "\n\n" + (prompt if prompt.strip() else "USER: (no message yet)")
        elif isinstance(prompt, list):
            prompt = [{"role": "user", "content": [{"text": _handoff}]}] + prompt

    # The app tells us which language to reply in via the request "locale" field (en/de/fr).
    _locale = _locale_preamble(payload)
    if _locale:
        if isinstance(prompt, str):
            prompt = _locale + "\n\n" + (prompt if prompt.strip() else "USER: (no message yet)")
        elif isinstance(prompt, list):
            prompt = [{"role": "user", "content": [{"text": _locale}]}] + prompt

    # Emit an ephemeral status bit FIRST - BEFORE the model starts producing - so the app can
    # show a "thinking" indicator during the wait and hide it as soon as the message streams.
    yield {"status_event": _status_label(payload)}

    async for event in agent.stream_async(
        prompt,
    ):
        if not isinstance(event, dict) or "event" not in event:
            continue
        cbs = event["event"].get("contentBlockStart")
        if cbs is not None and not cbs.get("start"):
            continue
        yield event


if __name__ == "__main__":
    app.run()
