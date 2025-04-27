// Global variables
let scene, camera, renderer, clock;
let controls;

// Simulation data
let simulationData;
let currentFrameIndex = 0;
const droneObjects = new Map(); // Store THREE.js drone objects
let lastFrameUpdate = 0; // Track last frame update time
const FRAME_INTERVAL = 1000; // 1 second between position updates
const ROTOR_SPEED = 0.3; // Speed of rotor rotation

// Initialize the scene
async function init() {
    // Load simulation data
    try {
        const response = await fetch('settings_simple.json');
        simulationData = await response.json();
        console.log('Loaded simulation data:', simulationData);
    } catch (error) {
        console.error('Failed to load simulation data:', error);
        return;
    }
    
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x87ceeb); // Sky blue background

    // Create camera with better position for viewing
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(5, 8, 12);
    camera.lookAt(5, 2, 5);

    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    document.body.appendChild(renderer.domElement);

    // Add OrbitControls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 5;
    controls.maxDistance = 50;
    controls.maxPolarAngle = Math.PI / 2;

    // Add lights
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    // Create ground
    const groundGeometry = new THREE.PlaneGeometry(
        simulationData.world.size_x,
        simulationData.world.size_z
    );
    const groundMaterial = new THREE.MeshPhongMaterial({ color: 0x808080 });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(
        simulationData.world.size_x / 2,
        0,
        simulationData.world.size_z / 2
    );
    ground.receiveShadow = true;
    scene.add(ground);

    // Add coordinate axes helper
    const axesHelper = new THREE.AxesHelper(5);
    scene.add(axesHelper);

    // Create buildings
    createBuildings();

    // Create initial drones
    createDrones();

    // Initialize clock
    clock = new THREE.Clock();

    // Start animation loop
    renderFrame();
}

function createBuildings() {
    simulationData.buildings.forEach(building => {
        const geometry = new THREE.BoxGeometry(
            building.width,
            building.height,
            building.depth
        );
        const material = new THREE.MeshPhongMaterial({ color: 0x808080 });
        const mesh = new THREE.Mesh(geometry, material);
        
        mesh.position.set(
            building.x,
            building.height / 2,
            building.y // Note: y in JSON is z in Three.js
        );
        
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);
    });
}

function createDrones() {
    // Get initial drone positions from first frame
    const initialDrones = simulationData.frames[0].drones;
    
    // Create each drone
    Object.entries(simulationData.drones).forEach(([droneId, droneInfo]) => {
        const initialPos = initialDrones[droneId];
        const size = { x: 0.5, y: 0.2, z: 0.5 }; // Increased size for better visibility
        const color = droneInfo.team === 'red' ? 0xff0000 : 0x0000ff;
        
        // Create drone body
        const drone = new THREE.Group();
        
        // Main body with emissive material for better visibility
        const body = new THREE.Mesh(
        new THREE.BoxGeometry(size.x, size.y, size.z),
            new THREE.MeshPhongMaterial({ 
                color: color,
                emissive: droneInfo.team === 'red' ? 0x330000 : 0x000033,
                emissiveIntensity: 0.5
            })
    );
    body.castShadow = true;
    drone.add(body);
    
        // Add rotors
        const rotorSize = size.x * 0.3;
    const rotorHeight = size.y * 0.1;
        const rotorGeometry = new THREE.CylinderGeometry(
            rotorSize,
            rotorSize,
            rotorHeight,
            8
        );
        const rotorMaterial = new THREE.MeshPhongMaterial({ color: 0x666666 });
        
        // Add four rotors
    const rotorPositions = [
            { x: size.x/2, y: 0, z: size.z/2 },
            { x: -size.x/2, y: 0, z: size.z/2 },
            { x: size.x/2, y: 0, z: -size.z/2 },
            { x: -size.x/2, y: 0, z: -size.z/2 }
        ];
        
        rotorPositions.forEach(pos => {
            const rotor = new THREE.Mesh(rotorGeometry, rotorMaterial);
        rotor.position.set(pos.x, pos.y, pos.z);
        rotor.rotation.x = Math.PI/2;
        rotor.castShadow = true;
        drone.add(rotor);
    });
    
        // Set initial position
        drone.position.set(initialPos.x, initialPos.y, initialPos.z);
        
        // Store metadata
        drone.userData = {
            team: droneInfo.team,
            id: droneId,
            rotors: drone.children.slice(1) // All children except body
        };
        
        scene.add(drone);
        droneObjects.set(droneId, drone);
    });
}

function updateDronePositions() {
    const frame = simulationData.frames[currentFrameIndex];
    if (!frame) return;

    const currentTime = clock.getElapsedTime() * 1000; // Convert to milliseconds
    
    // Update rotor animations every frame (60 FPS)
    droneObjects.forEach(drone => {
        drone.userData.rotors.forEach((rotor, index) => {
            rotor.rotation.x += (index % 2 ? ROTOR_SPEED : -ROTOR_SPEED);
        });
    });
    
    // Only update drone positions if enough time has passed (1 second)
    if (currentTime - lastFrameUpdate >= FRAME_INTERVAL) {
        // Update all drone positions
        Object.entries(frame.drones).forEach(([droneId, droneData]) => {
            const drone = droneObjects.get(droneId);
            if (drone) {
                // Update position
                drone.position.set(droneData.x, droneData.y, droneData.z);
            }
        });

        // Update frame index and last update time
        currentFrameIndex = (currentFrameIndex + 1) % simulationData.frames.length;
        lastFrameUpdate = currentTime;
    }
}

function renderFrame() {
    // Update controls
    controls.update();
    
    // Update drone positions based on simulation data
    updateDronePositions();
    
    // Render scene
    renderer.render(scene, camera);
    
    // Request next frame
    requestAnimationFrame(renderFrame);
}

// Add window resize handler
window.addEventListener('resize', onWindowResize, false);

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// Start the initialization
init(); 