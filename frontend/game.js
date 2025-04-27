// Global variables
let scene, camera, renderer, clock;
let physicsWorld;
let rigidBodies = [];
let tmpTrans;

// Wait for the page to be fully loaded
window.addEventListener('load', function() {
    // Check if Ammo is ready
    if (window.Ammo) {
        init();
    } else {
        console.error('Ammo.js not loaded yet');
    }
});

// Initialize the scene
function init() {
    tmpTrans = new Ammo.btTransform();
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x87ceeb); // Sky blue background

    // Create camera
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 15, 30);
    camera.lookAt(0, 0, 0);

    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    document.body.appendChild(renderer.domElement);

    // Add lights
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(1, 1, 1);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    // Create ground
    const groundGeometry = new THREE.PlaneGeometry(100, 100);
    const groundMaterial = new THREE.MeshPhongMaterial({ color: 0x808080 });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);

    // Initialize physics
    setupPhysicsWorld();

    // Add ground to physics world
    const groundShape = new Ammo.btStaticPlaneShape(new Ammo.btVector3(0, 1, 0), 0);
    const groundTransform = new Ammo.btTransform();
    groundTransform.setIdentity();
    const groundMotionState = new Ammo.btDefaultMotionState(groundTransform);
    const groundRigidBody = new Ammo.btRigidBody(new Ammo.btRigidBodyConstructionInfo(0, groundMotionState, groundShape));
    physicsWorld.addRigidBody(groundRigidBody);

    // Initialize clock
    clock = new THREE.Clock();

    // Create drones further apart
    createDrone(new THREE.Vector3(-15, 5, 0), new THREE.Vector3(2, 0.5, 2), 1, 0xff0000, 'red');
    createDrone(new THREE.Vector3(15, 5, 0), new THREE.Vector3(2, 0.5, 2), 1, 0x0000ff, 'blue');

    // Setup collision detection
    const collisionCallback = setupCollisionDetection();
    physicsWorld.setInternalTickCallback(() => {
        physicsWorld.getDispatcher().dispatchAllCollisionPairs(
            physicsWorld.getPairCache().getOverlappingPairArray(),
            physicsWorld.getDispatcher(),
            collisionCallback
        );
    });

    // Start animation loop
    renderFrame();
}

// Create a physics world with customized gravity
function setupPhysicsWorld() {
    let collisionConfiguration = new Ammo.btDefaultCollisionConfiguration();
    let dispatcher = new Ammo.btCollisionDispatcher(collisionConfiguration);
    let overlappingPairCache = new Ammo.btDbvtBroadphase();
    let solver = new Ammo.btSequentialImpulseConstraintSolver();
    
    physicsWorld = new Ammo.btDiscreteDynamicsWorld(dispatcher, overlappingPairCache, solver, collisionConfiguration);
    physicsWorld.setGravity(new Ammo.btVector3(0, -2, 0)); // Lower gravity for drones
}

// Create a drone with physics properties
function createDrone(position, size, mass, color, team) {
    // Three.js visualization
    let drone = new THREE.Group();
    
    // Create drone body (center cube)
    let body = new THREE.Mesh(
        new THREE.BoxGeometry(size.x, size.y, size.z),
        new THREE.MeshPhongMaterial({color: color})
    );
    body.castShadow = true;
    drone.add(body);
    
    // Create four arms
    const armLength = size.x * 0.7;
    const armWidth = size.x * 0.1;
    const armHeight = size.y * 0.2;
    const armGeometry = new THREE.BoxGeometry(armLength, armHeight, armWidth);
    const armMaterial = new THREE.MeshPhongMaterial({color: 0x333333});
    
    // Front right arm
    let armFR = new THREE.Mesh(armGeometry, armMaterial);
    armFR.position.set(armLength/2, 0, armLength/2);
    armFR.rotation.y = Math.PI/4;
    armFR.castShadow = true;
    drone.add(armFR);
    
    // Front left arm
    let armFL = new THREE.Mesh(armGeometry, armMaterial);
    armFL.position.set(-armLength/2, 0, armLength/2);
    armFL.rotation.y = -Math.PI/4;
    armFL.castShadow = true;
    drone.add(armFL);
    
    // Back right arm
    let armBR = new THREE.Mesh(armGeometry, armMaterial);
    armBR.position.set(armLength/2, 0, -armLength/2);
    armBR.rotation.y = -Math.PI/4;
    armBR.castShadow = true;
    drone.add(armBR);
    
    // Back left arm
    let armBL = new THREE.Mesh(armGeometry, armMaterial);
    armBL.position.set(-armLength/2, 0, -armLength/2);
    armBL.rotation.y = Math.PI/4;
    armBL.castShadow = true;
    drone.add(armBL);
    
    // Create rotors
    const rotorRadius = size.x * 0.3;
    const rotorHeight = size.y * 0.1;
    const rotorGeometry = new THREE.CylinderGeometry(rotorRadius, rotorRadius, rotorHeight, 8);
    const rotorMaterial = new THREE.MeshPhongMaterial({color: 0x666666});
    
    // Add rotors at the end of each arm
    const rotorPositions = [
        {x: armLength * 0.7, y: 0, z: armLength * 0.7},  // FR
        {x: -armLength * 0.7, y: 0, z: armLength * 0.7}, // FL
        {x: armLength * 0.7, y: 0, z: -armLength * 0.7}, // BR
        {x: -armLength * 0.7, y: 0, z: -armLength * 0.7} // BL
    ];
    
    rotorPositions.forEach((pos) => {
        let rotor = new THREE.Mesh(rotorGeometry, rotorMaterial);
        rotor.position.set(pos.x, pos.y, pos.z);
        rotor.rotation.x = Math.PI/2;
        rotor.castShadow = true;
        drone.add(rotor);
    });
    
    scene.add(drone);
    
    // Ammo.js physics - use a compound shape for better physics
    let transform = new Ammo.btTransform();
    transform.setIdentity();
    transform.setOrigin(new Ammo.btVector3(position.x, position.y, position.z));
    
    let motionState = new Ammo.btDefaultMotionState(transform);
    
    // Create compound shape
    let compoundShape = new Ammo.btCompoundShape();
    
    // Add central body collision box
    let bodyShape = new Ammo.btBoxShape(new Ammo.btVector3(size.x/2, size.y/2, size.z/2));
    let bodyTransform = new Ammo.btTransform();
    bodyTransform.setIdentity();
    compoundShape.addChildShape(bodyTransform, bodyShape);
    
    let localInertia = new Ammo.btVector3(0, 0, 0);
    compoundShape.calculateLocalInertia(mass, localInertia);
    
    let bodyInfo = new Ammo.btRigidBodyConstructionInfo(mass, motionState, compoundShape, localInertia);
    let physicsBody = new Ammo.btRigidBody(bodyInfo);
    
    physicsBody.setActivationState(4); // DISABLE_DEACTIVATION
    physicsBody.setDamping(0.5, 0.5); // Increase damping for more stable flight
    
    physicsWorld.addRigidBody(physicsBody);
    
    drone.userData.physicsBody = physicsBody;
    physicsBody.threeObject = drone;
    
    drone.userData.team = team;
    drone.userData.health = 100;
    drone.userData.tag = "drone";
    
    // Store rotors for animation
    drone.userData.rotors = rotorPositions.map((pos, index) => {
        return drone.children[index + 5]; // +5 because body and arms come first
    });
    
    rigidBodies.push(drone);
    
    return drone;
}

// Create a projectile
function createProjectile(position, direction, team) {
    const size = new THREE.Vector3(0.2, 0.2, 0.2);
    const mass = 0.1;
    const color = team === 'red' ? 0xff0000 : 0x0000ff;
    
    const projectile = createDrone(position, size, mass, color, team);
    projectile.userData.tag = "projectile";
    
    // Apply initial velocity
    const physicsBody = projectile.userData.physicsBody;
    const velocity = new Ammo.btVector3(
        direction.x * 20,
        direction.y * 20,
        direction.z * 20
    );
    physicsBody.setLinearVelocity(velocity);
    
    return projectile;
}

// Handle drone hit by projectile
function handleDroneHit(drone, projectile) {
    drone.userData.health -= 20;
    if (drone.userData.health <= 0) {
        scene.remove(drone);
        physicsWorld.removeRigidBody(drone.userData.physicsBody);
        const index = rigidBodies.indexOf(drone);
        if (index > -1) {
            rigidBodies.splice(index, 1);
        }
    }
    
    scene.remove(projectile);
    physicsWorld.removeRigidBody(projectile.userData.physicsBody);
    const index = rigidBodies.indexOf(projectile);
    if (index > -1) {
        rigidBodies.splice(index, 1);
    }
}

// Handle drone-drone collision
function handleDroneCollision(drone1, drone2) {
    // Simple collision response - just bounce off each other
    const physicsBody1 = drone1.userData.physicsBody;
    const physicsBody2 = drone2.userData.physicsBody;
    
    const velocity1 = physicsBody1.getLinearVelocity();
    const velocity2 = physicsBody2.getLinearVelocity();
    
    physicsBody1.setLinearVelocity(velocity2);
    physicsBody2.setLinearVelocity(velocity1);
}

// Setup collision detection
function setupCollisionDetection() {
    let cbContactResult = new Ammo.ConcreteContactResultCallback();
    
    cbContactResult.addSingleResult = function(cp, colObj0Wrap, partId0, index0, colObj1Wrap, partId1, index1) {
        let contactPoint = Ammo.wrapPointer(cp, Ammo.btManifoldPoint);
        
        // Get the colliding objects
        let colObj0 = Ammo.wrapPointer(colObj0Wrap.getCollisionObject(), Ammo.btCollisionObject);
        let colObj1 = Ammo.wrapPointer(colObj1Wrap.getCollisionObject(), Ammo.btCollisionObject);
        
        // Get the 3D objects
        let object0 = colObj0.threeObject;
        let object1 = colObj1.threeObject;
        
        // Handle drone-projectile collision
        if (object0?.userData?.tag === "drone" && object1?.userData?.tag === "projectile") {
            handleDroneHit(object0, object1);
        }
        else if (object1?.userData?.tag === "drone" && object0?.userData?.tag === "projectile") {
            handleDroneHit(object1, object0);
        }
        
        // Handle drone-drone collision
        if (object0?.userData?.tag === "drone" && object1?.userData?.tag === "drone") {
            handleDroneCollision(object0, object1);
        }
        
        return 0;
    };
    
    return cbContactResult;
}

// Update AI drones with more aggressive movement
function updateDroneAI() {
    rigidBodies.forEach((drone1, index) => {
        if (drone1.userData.tag === "drone") {
            let closestEnemy = null;
            let minDistance = Infinity;
            
            rigidBodies.forEach((drone2) => {
                if (drone2.userData.tag === "drone" && 
                    drone2.userData.team !== drone1.userData.team) {
                    const distance = drone1.position.distanceTo(drone2.position);
                    if (distance < minDistance) {
                        minDistance = distance;
                        closestEnemy = drone2;
                    }
                }
            });
            
            if (closestEnemy) {
                const direction = new THREE.Vector3();
                direction.subVectors(closestEnemy.position, drone1.position);
                direction.normalize();
                
                // Add slight vertical oscillation
                const time = Date.now() * 0.001;
                const heightOffset = Math.sin(time * 2) * 0.3;
                
                const controls = {
                    right: direction.x * 2,  // Increased force
                    up: direction.y + heightOffset,
                    forward: -direction.z * 2,  // Increased force
                    left: 0,
                    down: 0,
                    backward: 0
                };
                
                controlDrone(drone1, controls);
                
                // Rotate drone to face direction of travel
                const targetRotation = Math.atan2(direction.x, direction.z);
                const currentRotation = drone1.rotation.y;
                drone1.rotation.y += (targetRotation - currentRotation) * 0.1;
            }
        }
    });
}

// Control drone
function controlDrone(drone, controls) {
    let physicsBody = drone.userData.physicsBody;
    
    // Apply forces for movement
    let force = new Ammo.btVector3(
        controls.right - controls.left || 0,
        controls.up - controls.down || 0,
        controls.backward - controls.forward || 0
    );
    
    // Scale the force
    force.op_mul(20);
    
    physicsBody.applyForce(force, new Ammo.btVector3(0, 0, 0));
}

// Render function
function renderFrame() {
    let deltaTime = clock.getDelta();
    
    // Update AI drones
    updateDroneAI();
    
    // Update physics
    physicsWorld.stepSimulation(deltaTime, 10);
    
    // Update all objects
    for (let i = 0; i < rigidBodies.length; i++) {
        let objThree = rigidBodies[i];
        let objPhys = objThree.userData.physicsBody;
        let ms = objPhys.getMotionState();
        
        if (ms) {
            ms.getWorldTransform(tmpTrans);
            let p = tmpTrans.getOrigin();
            let q = tmpTrans.getRotation();
            
            objThree.position.set(p.x(), p.y(), p.z());
            objThree.quaternion.set(q.x(), q.y(), q.z(), q.w());
            
            // Animate rotors if this is a drone
            if (objThree.userData.rotors) {
                objThree.userData.rotors.forEach((rotor, index) => {
                    rotor.rotation.x += (index % 2 ? 0.5 : -0.5); // Alternate rotation directions
                });
            }
        }
    }
    
    // Render scene
    renderer.render(scene, camera);
    
    // Request next frame
    requestAnimationFrame(renderFrame);
}

// Start the initialization
init(); 