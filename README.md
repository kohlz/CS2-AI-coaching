# CS2-AI-coaching
AI coaching system 
# CS2-AI-coaching

## Project Overview
This project aims to build an AI coaching assistant for Counter-Strike 2 (CS2). Instead of taking over gameplay or directly controlling player actions, the system is designed to analyze the current game situation and provide context-aware recommendations that help players improve positioning and in-game decision-making.

Our long-term goal is to build a prototype that can interpret game-state information and generate useful coaching suggestions such as whether a player should rotate, hold a position, reposition, or play more safely.

## Motivation
Counter-Strike 2 has a steep learning curve. Players need to make fast decisions based on positioning, teammate information, utility usage, timing, and incomplete knowledge of the enemy team. Human coaching can help, but it is often expensive and not always available.

We are interested in exploring whether an AI-based system can provide useful and accessible coaching support by analyzing game situations and offering recommendations during or after gameplay.

## Project Scope
To keep the project feasible within one semester, we are currently limiting the scope to one competitive map.

Rather than trying to solve every aspect of the game at once, we want to build a focused prototype that can handle a smaller set of meaningful in-game situations.

## Problem Definition
Our project is not an automated gameplay agent.

Instead, the system focuses on:
1. extracting useful information about the current game state
2. representing the situation in a structured way
3. generating text-based coaching recommendations

Examples of recommendations may include:
- rotate to the other site
- hold current position
- reposition to a safer angle
- play slower and wait for teammate support
- avoid overcommitting because of limited time or missing information

## Current Technical Direction
At this stage, we are thinking about the project in multiple steps.

### Step 1: Game-State Understanding
We first need a way to extract useful state information from the game. Our current ideas include:
- using demo-based data where structured information may already be available
- using visual input such as minimap or player point-of-view
- exploring computer vision methods such as CNN-based localization or YOLO-style detection to identify meaningful spatial or situational information

### Step 2: Situation Representation
After extracting useful information, we want to represent the game state in a simpler form that the decision-making system can use. Possible state information includes:
- player position
- teammate locations
- deployed utility
- round timer
- likely enemy presence in map regions
- recent key events, such as a teammate being eliminated

### Step 3: Recommendation Generation
Once the state is represented, the system will generate text-based recommendations. Our first prototype will likely use a simpler state-based or classification-based method before attempting a more advanced probabilistic or belief-state approach.

### Possible Extension
If time allows, we may extend the system with a more advanced uncertainty-aware component inspired by partially observable settings, where the system estimates the probability of enemy presence across different map areas and uses that to improve recommendations.

## Planned Inputs
The current planned inputs to the system include:
- minimap information
- player point-of-view information
- teammate locations
- deployed utilities (such as smoke grenades)
- round timer
- important round events

## Planned Outputs
The output of the system will be text-based coaching suggestions.

Some example outputs include:
- "Rotate to B now."
- "Hold your current angle."
- "Fall back and wait for support."
- "Do not rotate yet; there is not enough confirmation."
- "Play safer because time is low and utility is limited."

## Example Use Case
Suppose a teammate is eliminated at a different bomb site late in the round. The system could use remaining time, teammate positioning, and available information to estimate whether a site hit is likely and suggest whether the player should rotate.

More generally, the goal is to produce recommendations that are useful, timely, and understandable to the player.

## Repository Structure
This repository is currently organized as follows:

- `README.md`  
  Project overview, setup notes, repository organization, and current direction.

- `docs/`  
  Check-in summary, project tracker, approach notes, and planning documents.

- `resources/`  
  References, demos, papers, and external resources related to the project.

- `src/`  
  Source code for the project implementation.

- `data/`  
  Placeholder directory for demo data, extracted features, or sample scenarios.

## Current Progress
So far, we have:
- completed the initial project proposal
- clarified that the project is an AI coaching assistant rather than an automated agent
- limited the scope to one map
- created the GitHub repository
- started organizing project documentation and tracking
- discussed multiple implementation directions, including demo-based data, visual methods, and recommendation generation

## Next Steps
Our next planned steps are:
1. organize the repository and tracking materials
2. collect and document useful papers, demos, and technical resources
3. decide on the first practical input pipeline
4. define the first version of the state representation
5. implement an initial prototype for recommendation generation
6. evaluate whether the first prototype is strong enough before adding a more advanced uncertainty-aware component

## Installation
This repository is still in the early project stage, so the full implementation is not yet available.

Planned environment:
- Python 3.x

Planned dependencies may include:
- numpy
- pandas
- opencv-python
- pytorch
- ultralytics / YOLO-related packages
- matplotlib

Detailed setup and run instructions will be added as implementation progresses.

## Usage
Usage instructions will be added once the first prototype is implemented.

Our eventual goal is for a user to:
1. provide game-state data or input frames
2. run the analysis pipeline
3. receive a text-based coaching recommendation

## References and Related Work
We are currently drawing inspiration from prior work and resources related to Counter-Strike analysis, including:
- Learning to Move Like Professional Counter-Strike Players
- Optimal Team Economic Decisions in Counter-Strike
- professional demos from public sources such as HLTV
- additional demos from our own gameplay

A more complete resource list will be maintained in the `resources/` folder.

## Disclaimer
This project is intended for academic use as part of a course project. The system is designed as a coaching and recommendation tool, not as a gameplay automation or cheating tool.
