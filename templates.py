from llama_index.core.prompts import PromptTemplate

DM_ANALYSIS_TEMPLATE = PromptTemplate(
    template="""
You are a master Dungeon Master analyzing the current story state.

CURRENT STATE:
{running_story}

Analyze and provide:
1. Next scene type (COMBAT, EXPLORATION, DIALOGUE, PUZZLE, DRAMATIC)
2. Suggested genre (FANTASY, SCIFI, HORROR, MYSTERY, ADVENTURE)
3. Current narrative tension (1-10)
4. Key story elements
"""
)

SEGMENT_TEMPLATE = PromptTemplate(
    template="""
Generate the next story segment based on the current analysis.

ANALYSIS:
{analysis}

Create a compelling segment with:
1. Descriptive plot
2. Available player actions
3. Appropriate scene type
4. Consistent genre
"""
)

CHARACTER_TEMPLATE = PromptTemplate(
    template="""
Create or update a character based on the world details.

WORLD DETAIL:
{world_detail}

Define the character's:
1. Name
2. Role in the story
3. Current motivation
4. Growth opportunities
5. Possible choices
"""
)

WORLD_TEMPLATE = PromptTemplate(
    template="""
Enhance the world with vivid details.

SEGMENT:
{segment}

Describe:
1. Current location
2. Environmental details
3. Hidden secrets
"""
)

TWIST_TEMPLATE = PromptTemplate(
    template="""
Design a surprising yet logical plot twist.

ANALYSIS:
{analysis}

Create:
1. A major revelation
2. Its impact on the story
3. Key elements affected
"""
)

IMPACT_TEMPLATE = PromptTemplate(
    template="""
Evaluate the impact of the chosen action.

CHOICE:
{choice}

Determine the consequences:
"""
)