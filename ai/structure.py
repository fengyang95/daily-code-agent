from pydantic import BaseModel, Field, field_validator
import re


class Structure(BaseModel):
    tldr: str = Field(description="generate a too long; didn't read summary")
    motivation: str = Field(description="describe the motivation in this paper")
    method: str = Field(description="method of this paper")
    result: str = Field(description="result of this paper")
    conclusion: str = Field(description="conclusion of this paper")

    topics: str = Field(description="topics of this paper. Can only include one or more of the following candidates: "
                                    "[code agent,agent analysis,agentic reinforcement learning,llm LLM,llm,swe application,swe benchmark,other topic]")
