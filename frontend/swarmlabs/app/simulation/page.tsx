"use client"

import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import Link from "next/link"
import { ArrowLeft, Pause, Play, RefreshCw } from "lucide-react"

type Drone = {
  id: number
  x: number
  y: number
  vx: number
  vy: number
  type: "red" | "blue"
}

export default function SimulationPage() {
  const [isRunning, setIsRunning] = useState(true)
  const [drones, setDrones] = useState<Drone[]>([])
  const [stats, setStats] = useState({
    redRemaining: 0,
    blueRemaining: 0,
    timeElapsed: 0,
  })

  // Initialize simulation
  useEffect(() => {
    // In a real app, we would get these values from the form
    const redDrones = 5
    const blueDrones = 5

    const initialDrones: Drone[] = []

    // Create red drones (attackers)
    for (let i = 0; i < redDrones; i++) {
      initialDrones.push({
        id: i,
        x: Math.random() * 100,
        y: Math.random() * 20 + 10,
        vx: (Math.random() - 0.5) * 2,
        vy: (Math.random() - 0.5) * 2,
        type: "red",
      })
    }

    // Create blue drones (defenders)
    for (let i = 0; i < blueDrones; i++) {
      initialDrones.push({
        id: i + redDrones,
        x: Math.random() * 100,
        y: Math.random() * 20 + 70,
        vx: (Math.random() - 0.5) * 2,
        vy: (Math.random() - 0.5) * 2,
        type: "blue",
      })
    }

    setDrones(initialDrones)
    setStats({
      redRemaining: redDrones,
      blueRemaining: blueDrones,
      timeElapsed: 0,
    })
  }, [])

  // Simulation loop
  useEffect(() => {
    if (!isRunning) return

    const interval = setInterval(() => {
      setDrones((prevDrones) => {
        // Create a copy of the drones array
        const newDrones = [...prevDrones]

        // Move each drone
        for (let i = 0; i < newDrones.length; i++) {
          const drone = newDrones[i]

          // Update position
          drone.x += drone.vx
          drone.y += drone.vy

          // Bounce off walls
          if (drone.x < 0 || drone.x > 100) {
            drone.vx = -drone.vx
            drone.x = Math.max(0, Math.min(100, drone.x))
          }

          if (drone.y < 0 || drone.y > 100) {
            drone.vy = -drone.vy
            drone.y = Math.max(0, Math.min(100, drone.y))
          }

          // Add some randomness to movement
          if (Math.random() < 0.05) {
            drone.vx += (Math.random() - 0.5) * 0.5
            drone.vy += (Math.random() - 0.5) * 0.5

            // Limit velocity
            const speed = Math.sqrt(drone.vx * drone.vx + drone.vy * drone.vy)
            if (speed > 2) {
              drone.vx = (drone.vx / speed) * 2
              drone.vy = (drone.vy / speed) * 2
            }
          }

          // For blue drones, seek nearest red drone
          if (drone.type === "blue") {
            const redDrones = newDrones.filter((d) => d.type === "red")
            if (redDrones.length > 0) {
              // Find closest red drone
              let closestDrone = redDrones[0]
              let closestDistance = Math.sqrt(
                Math.pow(drone.x - closestDrone.x, 2) + Math.pow(drone.y - closestDrone.y, 2),
              )

              for (let j = 1; j < redDrones.length; j++) {
                const distance = Math.sqrt(
                  Math.pow(drone.x - redDrones[j].x, 2) + Math.pow(drone.y - redDrones[j].y, 2),
                )

                if (distance < closestDistance) {
                  closestDistance = distance
                  closestDrone = redDrones[j]
                }
              }

              // Move toward closest red drone
              const dx = closestDrone.x - drone.x
              const dy = closestDrone.y - drone.y
              const distance = Math.sqrt(dx * dx + dy * dy)

              if (distance > 0) {
                drone.vx += (dx / distance) * 0.1
                drone.vy += (dy / distance) * 0.1
              }

              // Check for collision (capture)
              if (distance < 3) {
                // Mark red drone for removal
                const index = newDrones.findIndex((d) => d.id === closestDrone.id)
                if (index !== -1) {
                  newDrones.splice(index, 1)
                }
              }
            }
          }
        }

        return newDrones
      })

      // Update stats
      setStats((prev) => {
        const redRemaining = drones.filter((d) => d.type === "red").length
        const blueRemaining = drones.filter((d) => d.type === "blue").length

        return {
          redRemaining,
          blueRemaining,
          timeElapsed: prev.timeElapsed + 0.1,
        }
      })
    }, 100)

    return () => clearInterval(interval)
  }, [isRunning, drones])

  const resetSimulation = () => {
    // Reset to initial state
    setIsRunning(false)

    // In a real app, we would get these values from the form
    const redDrones = 5
    const blueDrones = 5

    const initialDrones: Drone[] = []

    // Create red drones (attackers)
    for (let i = 0; i < redDrones; i++) {
      initialDrones.push({
        id: i,
        x: Math.random() * 100,
        y: Math.random() * 20 + 10,
        vx: (Math.random() - 0.5) * 2,
        vy: (Math.random() - 0.5) * 2,
        type: "red",
      })
    }

    // Create blue drones (defenders)
    for (let i = 0; i < blueDrones; i++) {
      initialDrones.push({
        id: i + redDrones,
        x: Math.random() * 100,
        y: Math.random() * 20 + 70,
        vx: (Math.random() - 0.5) * 2,
        vy: (Math.random() - 0.5) * 2,
        type: "blue",
      })
    }

    setDrones(initialDrones)
    setStats({
      redRemaining: redDrones,
      blueRemaining: blueDrones,
      timeElapsed: 0,
    })

    // Restart simulation after a short delay
    setTimeout(() => setIsRunning(true), 500)
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-background to-secondary/30 p-4">
      <div className="container mx-auto">
        <div className="mb-4 flex items-center justify-between">
          <Link href="/" className="flex items-center text-sm font-medium text-muted-foreground hover:text-foreground">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Parameters
          </Link>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => setIsRunning(!isRunning)}>
              {isRunning ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              {isRunning ? "Pause" : "Resume"}
            </Button>
            <Button variant="outline" size="sm" onClick={resetSimulation}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Reset
            </Button>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-4">
          <Card className="md:col-span-3">
            <CardHeader className="pb-2">
              <CardTitle>Drone Swarm Simulation</CardTitle>
              <CardDescription>Visualizing drone swarm behavior in real-time</CardDescription>
            </CardHeader>
            <CardContent>
              <div
                className="relative h-[500px] w-full overflow-hidden rounded-md border bg-black/10 backdrop-blur-sm"
                style={{ position: "relative" }}
              >
                {drones.map((drone) => (
                  <div
                    key={drone.id}
                    className={`drone ${drone.type === "red" ? "red-drone" : "blue-drone"}`}
                    style={{
                      left: `${drone.x}%`,
                      top: `${drone.y}%`,
                    }}
                  />
                ))}
              </div>
            </CardContent>
          </Card>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle>Simulation Stats</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Red Drones:</span>
                    <span className="font-medium text-destructive">{stats.redRemaining}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Blue Drones:</span>
                    <span className="font-medium text-primary">{stats.blueRemaining}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Time Elapsed:</span>
                    <span className="font-medium">{stats.timeElapsed.toFixed(1)}s</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle>Objective</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Drone Swarm Removal: Blue defensive drones are attempting to neutralize all red attacking drones.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle>Status</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex items-center">
                    <div
                      className={`mr-2 h-2 w-2 rounded-full ${stats.redRemaining === 0 ? "bg-green-500" : "bg-amber-500"}`}
                    ></div>
                    <span className="text-sm">{stats.redRemaining === 0 ? "Mission Complete" : "In Progress"}</span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {stats.redRemaining === 0
                      ? `All red drones neutralized in ${stats.timeElapsed.toFixed(1)} seconds`
                      : `${stats.redRemaining} red drones remaining`}
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </main>
  )
}
