# Atlas Product Specification

Atlas is a personal operating system for one user: Trang Nguyen.

Atlas is not primarily a chatbot, note search engine, research assistant, or
RAG application. Those may be interfaces or implementation techniques, but the
product goal is clarity about:

- what Trang knows
- what Trang is doing
- what Trang should do next
- what has changed
- why decisions were made
- what is important right now

The user is the primary entity. All indexed information exists to support
understanding, planning, retrieval, and decision-making for that user.

## Core Philosophy

Atlas should think in entities, not files.

Files are storage. The actual objects Atlas manages are:

- Projects
- Tasks
- Deliverables
- Decisions
- Goals
- Concepts
- Research Topics
- Books
- Courses
- Repositories
- Journal Entries
- People
- Events
- Files

Atlas should identify entities first and source documents second. Questions
should be answered by reasoning over entities and relationships, not merely by
retrieving document chunks.

## Core Capabilities

### Knowledge Retrieval

Atlas should answer:

- What is X?
- What do I know about X?
- Summarize X.
- What have I learned about X?
- What sources discuss X?
- What conclusions have I reached?

Responses should include:

- a synthesized answer
- supporting evidence
- source references
- related concepts

### Project Awareness

Atlas must maintain awareness of:

- active projects
- inactive projects
- project phases
- milestones
- deliverables
- blockers
- priorities

Atlas should answer:

- What projects are active?
- What changed recently?
- What remains unfinished?
- What is blocked?
- Which project needs attention?

### Task Management

Atlas should understand:

- completed tasks
- incomplete tasks
- overdue tasks
- dependencies
- priorities

Atlas should answer:

- What should I work on next?
- What is my highest priority task?
- What tasks are blocked?
- What tasks remain?

When asked what to do next, Atlas should:

1. Identify active projects.
2. Identify each relevant project's current phase.
3. Identify incomplete tasks.
4. Identify blockers.
5. Recommend one specific action.

Atlas must not answer with a project goal when the user requested a task.

### Decision Memory

Atlas should store and retrieve:

- decisions
- reasoning
- alternatives considered
- consequences

Atlas should answer:

- Why did I choose this?
- When was this decided?
- What alternatives were rejected?

### Learning Assistant

Atlas should understand:

- courses
- research topics
- study plans
- reading plans
- knowledge gaps

Atlas should answer:

- What should I study next?
- What concepts am I missing?
- What prerequisites have I not learned?
- What are the next milestones?

### Research Assistant

Atlas should support:

- concept exploration
- literature tracking
- research journaling
- idea synthesis
- question generation

Atlas should answer:

- What remains unknown?
- What should I investigate next?
- What papers are relevant?
- What patterns exist across sources?

### Personal Memory

Atlas should maintain awareness of:

- goals
- interests
- preferences
- recurring themes
- long-term projects

Atlas should answer:

- What am I focusing on lately?
- What have I neglected?
- What patterns appear in my work?

### Repository Awareness

Atlas should understand repositories as entities.

For each repository, Atlas should track:

- purpose
- status
- recent activity
- technologies
- open work
- known issues

Atlas should answer:

- What changed today?
- What repository is most active?
- Which projects are stalled?
- What did I work on this week?

### Local File Awareness

Atlas should understand:

- project folders
- important files
- documentation
- notes
- assets

Atlas should answer:

- Where is X?
- What files relate to X?
- What changed recently?

### Calendar and Event Awareness

When calendar or event data is available, Atlas should track:

- upcoming deadlines
- meetings
- exams
- milestones

Atlas should answer:

- What is due soon?
- What should I prepare for?
- What deadlines are approaching?

### Weekly and Monthly Reviews

Atlas should generate:

- accomplishments
- completed work
- unfinished work
- neglected projects
- upcoming priorities

Atlas should answer:

- What did I accomplish this week?
- What should I focus on next week?
- What am I ignoring?

## Entity Model

Atlas should gradually evolve toward a structured knowledge graph.

Primary entity types:

- Project
- Task
- Deliverable
- Goal
- Decision
- Concept
- Research Topic
- Book
- Course
- Repository
- Journal Entry
- Person
- Event
- File

Core relationships:

- Project -> Task
- Project -> Deliverable
- Project -> Goal
- Task -> Dependency
- Decision -> Project
- Concept -> Research Topic
- Book -> Concept
- Repository -> Project
- Journal Entry -> Project

Atlas should reason over these relationships.

## Response Requirements

Priorities:

1. Accuracy
2. Context awareness
3. Actionability
4. Traceability
5. Completeness

Every response should:

- answer the question directly
- identify relevant entities
- connect related information
- provide evidence
- surface missing information
- suggest next actions when appropriate

## Long-Term Vision

Atlas should become a personal operating system that integrates:

- Obsidian
- local files
- repositories
- tasks
- calendar
- journals
- research notes
- project documentation

The goal is not conversation. The goal is clarity.
