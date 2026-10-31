# Research 642: Mozilla Human-Scale AI Incubator — program digest

> **Source:** Mozilla Foundation Human-Scale AI Incubator — [overview](https://www.mozillafoundation.org/en/what-we-do/grantmaking/incubator/human-scale-ai/), [call for proposals](https://www.mozillafoundation.org/en/what-we-do/grantmaking/incubator/human-scale-ai/call-for-proposals/), [anchor projects](https://www.mozillafoundation.org/en/what-we-do/grantmaking/incubator/human-scale-ai/cohort/), [FAQs](https://www.mozillafoundation.org/en/what-we-do/grantmaking/incubator/human-scale-ai/call-for-proposals/faqs/) (FAQ last updated 2026-10-15)
> **Date:** 2026-10-31
> **Status:** Active — grant preparation runs in the private repos (pointer below)
> **Classification:** Public — program facts only. Eligibility verdicts, per-repo matrices, frictions, consult questions, and timeline are **grant strategy** and live in the private repos, not here.

---

## TL;DR

Mozilla Foundation's Human-Scale AI Incubator funds **seven projects, $300K + 12 months from April 2027, up to $500K/3yr total, non-dilutive**, for working AI technologies that are **Small**, **Intentional**, and **Tailored** (all three required). **Deadline 2026-11-20, 11:59 PM PST.** Two workspace candidates are being prepared in the private repos: the modelless code healer (`riir-refine`) and the open decision-serving stack (`riir-reflex` + `riir-instinct` surfaces). This note is the public-safe program digest; every strategic artifact is referenced in §5.

---

## 1. Program facts

| Item | Fact |
|---|---|
| Funder | Mozilla Foundation (US 501(c)(3)) — Incubator program |
| Award | $300K + 12 months intensive support, starting April 2027; strong progress → up to $200K + 2 more years (total $500K/3yr). Non-dilutive. |
| Cohort size | 7 projects + 3 pre-selected anchors (Tiles, Family Intelligence, DAIR) |
| Deadline | **Friday 2026-11-20, 11:59 PM PST (UTC-8)** = Nov 21 07:59 UTC. No extensions. |
| Info session | **Nov 12, 5 p.m. UTC** |
| Decision notice | Late February 2027; no individual updates |
| Contact | `incubator@mozillafoundation.org` |
| Eligibility | Must be legally able to receive funds from a US 501(c)(3); charitable purpose required; for-profit may apply with a charitability case; fiscal sponsorship recommended |
| Process | Initial Project Proposal → shortlist → full application; submitted forms are final; English |
| AI-use policy | AI tools OK for prep; submitted content must be the applicant's own original work and voice |

## 2. The three principles (ALL required)

1. **Small** — small models, less compute, local/low-power capable. Mozilla's calibration: *"models as small as 7B work great for text applications, while running a voice application locally might require up to 32B."*
2. **Intentional** — insight into how data is collected/stored/transmitted/used + meaningful agency (configure, adapt, fork, own, govern).
3. **Tailored** — fit-for-purpose for specific problems/communities/contexts; especially resource-constrained regions, people with disabilities, underserved populations; dialogue with the people served.

## 3. Hard exclusions (from the call)

- Concepts/wireframes without working technology
- Standalone governance/licensing/policy/research without a working tech component
- Technologies without users, contributors, or another meaningful early signal
- Generic chatbot applications
- Projects whose core functionality depends on a frontier model or a slimmed-down frontier model (e.g. Llama 1B/3B)

## 4. Evaluation criteria

Human-Scale AI expression · technical viability · problem-community clarity · product-community fit · impact and values · sustainability potential. Plus: cohort-level mix of geographies, communities, applications, technical approaches.

## 5. Where the strategy lives (private repos)

| Artifact | Home |
|---|---|
| Code-healer application draft + readiness playbook (eligibility matrices, verdicts, frictions, evidence inventory, consult-email draft, timeline) | `riir-refine/.proposals/025` + `riir-refine/.docs/01_orientation/mozilla_grant_readiness.md` + `riir-refine/.issues/190` (private repo) |
| Decision-stack application draft + readiness playbook | `riir-rethink/.proposals/005` + `riir-rethink/.docs/09_human_scale_ai/` + `riir-rethink/.issues/048` (private repo) |

## 6. Compliance notes

- **Mozilla AI-use policy:** agent-drafted material is internal preparation only; submitted answers are owner-voiced.
- **Numbering:** research 642 allocated via `.research/.highwater` (641 → 642, written back); dual_allocation_gate ran clean before allocation.
