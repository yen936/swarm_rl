from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, validator
import uvicorn
from typing import Optional
import os
from test_drone_rl import train_simple_agent

app = FastAPI(title="Drone Combat Training API", 
              description="API for training drone combat reinforcement learning agents")

class TrainingRequest(BaseModel):
    """Request model for training a drone combat agent"""
    num_blue_drones: int = Field(default=1, ge=1, le=5, description="Number of blue drones (controlled by the agent)")
    num_red_drones: int = Field(default=1, ge=1, le=5, description="Number of red drones (opponents)")
    total_timesteps: int = Field(default=10000, ge=1000, description="Total timesteps for training")

class TrainingResponse(BaseModel):
    """Response model for training completion"""
    status: str
    model_path: str
    training_steps: int
    message: str
    replay_path: Optional[str] = None
    replay_data: Optional[dict] = None

@app.post("/train", response_model=TrainingResponse)
async def train_agent(request: TrainingRequest):
    """
    Train a drone combat reinforcement learning agent
    
    This endpoint trains a PPO agent to control blue drones in combat against red drones.
    The number of drones on each team can be specified, up to a maximum of 5 per team.
    """
    try:
        print(f"Starting training with {request.num_blue_drones} blue drones vs {request.num_red_drones} red drones")
        print(f"Training for {request.total_timesteps} timesteps")
        
        # Train the agent
        model = train_simple_agent(
            total_timesteps=request.total_timesteps,
            num_blue_drones=request.num_blue_drones,
            num_red_drones=request.num_red_drones
        )
        
        # Prepare response
        response = TrainingResponse(
            status="success",
            model_path=request.save_path,
            training_steps=request.total_timesteps,
            message=f"Successfully trained agent with {request.num_blue_drones} blue drones vs {request.num_red_drones} red drones"
        )
        
        # Include replay data if it was recorded
        if request.record_replay and request.replay_path:
            replay_path = request.replay_path
            if os.path.exists(replay_path):
                try:
                    import json
                    with open(replay_path, 'r') as f:
                        replay_data = json.load(f)
                    response.replay_path = replay_path
                    response.replay_data = replay_data
                except Exception as e:
                    print(f"Error loading replay data: {str(e)}")
        
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
            "/docs": "API documentation"
        }
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
