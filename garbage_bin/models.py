from enum import Enum
from typing import List, Optional
from llama_index.core.bridge.pydantic import BaseModel, Field
import uuid

class GenreType(str, Enum):
    FANTASY = "fantasy"
    SCIFI = "sci-fi"
    HORROR = "horror"
    MYSTERY = "mystery"
    ADVENTURE = "adventure"

class SceneType(str, Enum):
    COMBAT = "combat"
    EXPLORATION = "exploration"
    DIALOGUE = "dialogue"
    PUZZLE = "puzzle"
    DRAMATIC = "dramatic"

class Character(BaseModel):
    name: str
    role: str
    motivation: str

class Segment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plot: str
    actions: List[str]
    scene_type: SceneType
    genre: GenreType

class DMAnalysis(BaseModel):
    next_scene_type: SceneType
    suggested_genre: GenreType
    narrative_tension: int = Field(ge=1, le=10)
    key_elements: List[str]

class PlotTwist(BaseModel):
    revelation: str
    impact: str
    key_elements: List[str]

class CharacterDevelopment(BaseModel):
    character: Character
    growth: str
    choices: List[str] 

class WorldDetail(BaseModel):
    location: str
    description: str
    secrets: List[str]

class Impact(BaseModel):
    consequences: List[str]
