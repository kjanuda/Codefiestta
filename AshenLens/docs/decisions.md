# Technical Decisions

## D-001 — Chosen Competition Track

### Decision
Sub-track 1A — Rich Answers, Not Just Text.

### Reason
The challenge requires answers that include relevant visual evidence
such as diagrams, tables, images, and scanned figures rather than
returning text alone.

The Ashen Era Archive contains a large number of visual assets and the
development question set contains multiple questions whose answers
depend directly on those visuals.

### Current Approach

The planned system will combine:

1. Text document retrieval
2. Visual asset retrieval
3. Vision-based image understanding
4. Grounded answer generation
5. Source and visual evidence presentation
## D-002 — Separate Text and Visual Indexes

### Decision

Text documents and standalone visual assets are normalized into
separate indexes.

### Reason

Sub-track 1A requires the system not only to retrieve textual evidence
but also to identify and display the actual relevant visual.

Keeping visual assets as first-class indexed objects allows the system
to retrieve figure plates, portraits, heraldry, and artifact
illustrations independently from textual chunks.

### Implementation

Text records:

storage/documents.jsonl

Visual assets:

storage/visual_assets.jsonl