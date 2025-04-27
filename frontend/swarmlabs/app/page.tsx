'use client'

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import Link from "next/link"
import { DrillIcon as Drone, Radar, Zap } from "lucide-react"

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-background to-secondary/30">
      <header className="container mx-auto py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Drone className="h-8 w-8 text-primary" />
            <h1 className="text-2xl font-bold tracking-tighter">SwarmLabs</h1>
          </div>
          <nav className="hidden md:block">
            <ul className="flex items-center gap-6">
              <li>
                <Link href="/" className="text-sm font-medium hover:text-primary">
                  Home
                </Link>
              </li>
              <li>
                <Link href="#" className="text-sm font-medium hover:text-primary">
                  About
                </Link>
              </li>
              <li>
                <Link href="#" className="text-sm font-medium hover:text-primary">
                  Research
                </Link>
              </li>
              <li>
                <Link href="#" className="text-sm font-medium hover:text-primary">
                  Contact
                </Link>
              </li>
            </ul>
          </nav>
          <Button variant="outline" size="sm">
            Login
          </Button>
        </div>
      </header>

      <section className="container mx-auto py-12 md:py-24">
        <div className="grid gap-6 md:grid-cols-2 md:gap-12">
          <div className="flex flex-col justify-center space-y-4">
            <div className="inline-flex items-center rounded-full border border-primary/20 bg-primary/10 px-3 py-1 text-sm text-primary">
              <Radar className="mr-1 h-3 w-3" />
              Advanced Drone Simulation
            </div>
            <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
              Simulate Drone Swarm Operations
            </h2>
            <p className="text-muted-foreground md:text-xl">
              Configure and visualize drone swarm behavior in various scenarios. Test defensive and offensive strategies
              in a controlled environment.
            </p>
          </div>
          <Card className="overflow-hidden border-primary/20 bg-background/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle>Simulation Parameters</CardTitle>
              <CardDescription>Configure your drone swarm simulation</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="lethality" className="text-sm font-medium">
                    Lethality
                  </Label>
                  <Switch id="lethality" />
                </div>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="latitude" className="text-sm font-medium">
                    Latitude
                  </Label>
                  <Input id="latitude" placeholder="e.g. 37.7749" type="number" step="0.0001" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="longitude" className="text-sm font-medium">
                    Longitude
                  </Label>
                  <Input id="longitude" placeholder="e.g. -122.4194" type="number" step="0.0001" />
                </div>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="red-drones" className="text-sm font-medium">
                    # of Red Drones
                  </Label>
                  <Input id="red-drones" type="number" min="1" max="20" placeholder="Max 20" defaultValue="5" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="blue-drones" className="text-sm font-medium">
                    # of Blue Drones
                  </Label>
                  <Input id="blue-drones" type="number" min="1" max="20" placeholder="Max 20" defaultValue="5" />
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="objective" className="text-sm font-medium">
                  Choose Objective
                </Label>
                <Select defaultValue="removal">
                  <SelectTrigger>
                    <SelectValue placeholder="Select objective" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="removal">Drone Swarm Removal</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
            <CardFooter>
              <Button 
                className="w-full"
                onClick={() => window.location.href = 'http://localhost:8010/'}
              >
                <Zap className="mr-2 h-4 w-4" />
                Run Simulation
              </Button>
            </CardFooter>
          </Card>
        </div>
      </section>

      <section className="container mx-auto py-12">
        <div className="grid gap-8 md:grid-cols-3">
          <Card className="border-primary/20 bg-background/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle>Advanced Algorithms</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                Our simulation uses cutting-edge swarm intelligence algorithms to model realistic drone behavior.
              </p>
            </CardContent>
          </Card>
          <Card className="border-primary/20 bg-background/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle>Real-time Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                Get instant feedback on swarm performance and effectiveness in various scenarios.
              </p>
            </CardContent>
          </Card>
          <Card className="border-primary/20 bg-background/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle>Strategic Insights</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                Develop and test counter-drone strategies in a controlled virtual environment.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>
    </main>
  )
}
