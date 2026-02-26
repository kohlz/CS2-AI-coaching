# Resources

## Papers / Related Work

### 1. Learning to Move Like Professional Counter-Strike Players
This paper is one of the main inspirations for our project. It shows how professional Counter-Strike behavior can be studied and modeled from gameplay data. It is helpful for thinking about player movement, positioning, and how game situations can be represented in a structured way.

### 2. Optimal Team Economic Decisions in Counter-Strike
This paper is useful because it shows how Counter-Strike gameplay can be studied through a more formal decision-making perspective. Although our project is focused more on coaching and positioning than economy, it helps us think about structured reasoning and decision support in the game.

---

## Demo / Data Sources

### 1. HLTV Demos
Public professional match demos from HLTV are one of our main planned data sources. These demos may help us study player movement, team behavior, timing, and map-specific situations.

### 2. Personal Gameplay Demos
We may also use our own gameplay demos as an additional source for simpler experiments, testing, or scenario examples.

---

## Technical Directions We Are Considering

### 1. Demo-Based Structured Data
One possible direction is to begin with demo-based data if we can extract structured information more easily. This may be a more practical first step for building an early prototype.

### 2. Vision-Based Input
Another direction is to use visual input such as minimap or point-of-view frames. This would require extracting useful game-state information from images or video.

### 3. CNN / YOLO for Scene Understanding
We have discussed using CNN-based localization or YOLO-style methods as possible tools for identifying meaningful map or scene information from visual inputs.

### 4. Recommendation / Coaching Layer
After obtaining useful state information, we want to build a recommendation component that can output coaching suggestions such as:
- rotate
- hold position
- reposition
- play safer
- wait for teammate support

### 5. Possible Extension: Belief-State / Probabilistic Reasoning
Based on professor feedback, a more advanced extension could involve estimating probabilities over enemy presence in different map areas under partial information. This would make the system more uncertainty-aware, but it is likely beyond the first prototype.

---

## Current Notes
At this stage, our main goal is not to finalize every technical detail, but to narrow down the most practical first prototype direction.

Our current focus is on:
- understanding what inputs are realistic
- deciding whether to start from demos or visual input
- defining useful recommendation categories
- keeping the scope manageable for one semester

---

## Planned Additions
This file will be expanded as we collect:
- more papers
- technical tutorials
- code libraries / tools
- demo parsing resources
- implementation references
