# SwarmLabs - Drone Swarm Simulation Platform

SwarmLabs is a modern web application for simulating and visualizing drone swarm operations. It consists of two main components:
1. A Next.js frontend application (this repository)
2. A simulation server that runs on port 8010

## Prerequisites

- Node.js 18.x or higher
- pnpm (recommended) or npm
- Python 3.8+ (for the simulation server)

## Setup Instructions

### 1. Frontend Setup (Next.js Application)

```bash
# Install dependencies
pnpm install

# Start the development server
pnpm dev
```

The frontend will be available at `http://localhost:3000`

### 2. Simulation Server Setup

The simulation server needs to be running on port 8010. You'll need to set this up separately.

1. Clone the simulation server repository (if separate)
2. Install Python dependencies
3. Start the simulation server on port 8010

## Usage

1. Start both servers:
   - Frontend: `pnpm dev` (runs on port 3000)
   - Simulation server: (should be running on port 8010)

2. Open `http://localhost:3000` in your browser

3. Configure simulation parameters:
   - Set the number of red and blue drones (1-20 each)
   - Adjust latitude and longitude if needed
   - Toggle lethality settings
   - Choose simulation objective

4. Click "Run Simulation" to start the simulation
   - This will redirect you to `http://localhost:8010`
   - The simulation visualization will begin automatically

## Project Structure

```
├── app/                    # Next.js app directory
│   ├── layout.tsx         # Root layout with theme provider
│   ├── page.tsx           # Main landing page
│   └── simulation/        # Simulation page components
├── components/            # Reusable UI components
│   ├── ui/               # Shadcn UI components
│   └── theme-provider.tsx # Dark/light theme provider
├── styles/               # Global styles
└── public/              # Static assets
```

## Features

- Real-time drone swarm visualization
- Configurable simulation parameters
- Dark/light theme support
- Responsive design
- Modern UI with Shadcn components

## Development

The project uses:
- Next.js 15.2.4
- React 19
- Shadcn UI components
- TailwindCSS for styling
- TypeScript for type safety

## Troubleshooting

1. **Simulation Server Not Responding**
   - Ensure the simulation server is running on port 8010
   - Check for any CORS issues
   - Verify network connectivity

2. **Frontend Issues**
   - Clear browser cache
   - Run `pnpm install` to ensure all dependencies are up to date
   - Check console for any JavaScript errors

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

[Add your license information here] 

python3 -m http.server 8052 ^ to run an actual run

