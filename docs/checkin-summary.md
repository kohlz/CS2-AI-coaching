# CS4100 Project Check-in Summary

## Project Title
CS2 AI Coaching System

## Problem / Goal
Our project aims to build an AI coaching assistant for Counter-Strike 2 (CS2). The system is not intended to play the game automatically or directly control player actions. Instead, it is designed to analyze the current game situation and provide context-aware recommendations to help players improve positioning and decision-making.

Our long-term goal is to build a prototype that can take in game-state information and generate useful text-based coaching suggestions.

## Current Scope
To keep the project feasible within one semester, we are currently limiting the scope to one competitive map.

Rather than trying to solve every part of the game at once, we want to focus on a smaller number of meaningful in-game situations and build a workable prototype.

## Current Technical Direction
Our current thinking is to build the project in stages.

First, we want to extract useful game-state information either from demo-based structured data or from visual input such as minimap / POV information.

Second, we want to represent the game situation in a simpler form using features such as:
- player position
- teammate locations
- deployed utility
- round timer
- likely enemy presence in important map areas
- recent key events

Third, we want to generate text-based coaching recommendations, such as whether the player should rotate, hold position, reposition, or play more safely.

For the first prototype, we are likely to start with a simpler state-based or classification-based approach before attempting a more advanced probabilistic or belief-state extension.

## Progress So Far
So far, we have:
- completed the initial project proposal
- clarified that the project is an AI coaching / recommendation system rather than an automated gameplay system
- limited the scope to one map
- created and updated the GitHub repository
- started organizing project documents and tracking materials
- discussed multiple implementation directions, including demo-based data, CNN / YOLO-style visual approaches, and recommendation generation

## Current Challenges
The main challenges we are currently thinking about are:
- deciding the most practical first input pipeline
- balancing project ambition with semester feasibility
- choosing the right level of complexity for the first prototype
- determining how to evaluate whether the recommendations are useful

## Next Steps
Our current next steps are:
1. organize the GitHub repository and project tracker
2. collect and document useful papers, demos, and technical resources
3. decide whether to begin with demo-based structured data, visual input, or a simplified state representation
4. define the first version of the recommendation categories
5. begin implementing the first prototype
6. refine the project direction after feedback from the TA check-in

## Questions for TA
1. Does our current one-map scope seem reasonable for the course project?
2. For the first prototype, would you recommend starting from demo-based structured data, visual input, or a simpler hand-designed state representation?
3. Would a simpler recommendation component be enough for the first prototype before adding a more advanced uncertainty-aware extension?
4. Does our current repo / tracker structure seem detailed enough for this stage?
