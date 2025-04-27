from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from typing import Optional
import os
from swarm_trainer import train_agent, train_agent_with_all_replays

app = FastAPI(
    title="Drone Combat Training API",
    description="API for training drone combat reinforcement learning agents",
)


class TrainingRequest(BaseModel):
    """Request model for training a drone combat agent"""

    num_blue_drones: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Number of blue drones (controlled by the agent)",
    )
    num_red_drones: int = Field(
        default=1, ge=1, le=5, description="Number of red drones (opponents)"
    )
    total_timesteps: int = Field(
        default=10000, ge=1000, description="Total timesteps for training"
    )
    save_path: str = Field(
        default="drone_ft_agent_model", description="Path to save the trained model"
    )
    record_replay: bool = Field(
        default=True, description="Whether to record replay data"
    )
    replay_path: str = Field(
        default="replay.json", description="Path to save replay data"
    )
    collect_all_replays: bool = Field(
        default=False, description="Whether to collect replays from all episodes"
    )
    max_steps: int = Field(
        default=1000, ge=100, description="Maximum steps per episode"
    )  

class TrainingResponse(BaseModel):
    """Response model for training completion"""

    status: str
    model_path: str
    training_steps: int
    message: str
    replay_path: Optional[str] = None
    replay_data: Optional[dict] = None
    all_replays_path: Optional[str] = None

@app.post("/train", response_model=TrainingResponse)
async def train_drone_agent(request: TrainingRequest):
    """
    Train a drone combat reinforcement learning agent

    This endpoint trains a PPO agent to control blue drones in combat against red drones.
    The number of drones on each team can be specified, up to a maximum of 5 per team.
    """
    try:
        print(
            f"Starting training with {request.num_blue_drones} blue drones vs {request.num_red_drones} red drones"
        )
        print(f"Training for {request.total_timesteps} timesteps")

        # Train the agent - either with all replays or just the final one
        if request.collect_all_replays:
            print("Collecting replays from all episodes in memory")
            model, all_replays = train_agent_with_all_replays(
                total_timesteps=request.total_timesteps,
                num_blue_drones=request.num_blue_drones,
                num_red_drones=request.num_red_drones,
                save_path=request.save_path,
                record_replay=True,  # Always record when collecting all replays
                max_steps=request.max_steps
            )
            replay_data = all_replays

        else:
            print("Collecting only final replay")
            model, replay_data = train_agent(
                total_timesteps=request.total_timesteps,
                num_blue_drones=request.num_blue_drones,
                num_red_drones=request.num_red_drones,
                save_path=request.save_path,
                record_replay=request.record_replay,
                replay_path=request.replay_path
            )

        # Prepare response
        response = TrainingResponse(
            status="success",
            model_path=request.save_path,
            training_steps=request.total_timesteps,
            message=f"Successfully trained agent with {request.num_blue_drones} blue drones vs {request.num_red_drones} red drones",
        )

        # Include replay data in the response
        if replay_data:
            # Only include the actual replay data if it's not too large
            # This prevents response size issues with many episodes
            estimated_size = len(str(replay_data))
            if estimated_size < 10000000:  # ~10MB limit
                response.replay_data = replay_data
                print(f"Including replay data in response (approx. {estimated_size/1000000:.2f} MB)")
            else:
                # If data is too large, just indicate it's too large
                print(f"Replay data too large for response (approx. {estimated_size/1000000:.2f} MB)")
                response.message += f". Replay data too large to include in response ({estimated_size/1000000:.2f} MB)."

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Drone Combat Training API",
        "version": "1.0.0",
        "description": "API for training drone combat reinforcement learning agents",
        "endpoints": {
            "/train": "Train a drone combat agent",
            "/docs": "API documentation",
        },
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
