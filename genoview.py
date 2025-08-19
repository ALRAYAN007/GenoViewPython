from pyray import (
    Vector2, Vector3, Vector4, Transform, Matrix, Camera3D, 
    Color, Rectangle, Model, ModelAnimation, Mesh, BoneInfo, 
    Texture, RenderTexture)
from raylib import *

import bvh
import quat
import numpy as np
import struct
import cffi
ffi = cffi.FFI()

#----------------------------------------------------------------------------------
# Camera
#----------------------------------------------------------------------------------

class Camera:

    def __init__(self):
        self.cam3d = Camera3D()
        self.cam3d.position = Vector3(2.0, 3.0, 5.0)
        self.cam3d.target = Vector3(-0.5, 1.0, 0.0)
        self.cam3d.up = Vector3(0.0, 1.0, 0.0)
        self.cam3d.fovy = 45.0
        self.cam3d.projection = CAMERA_PERSPECTIVE
        self.azimuth = 0.0
        self.altitude = 0.4
        self.distance = 4.0
        self.offset = Vector3Zero()
    
    def update(
        self,
        target,
        azimuthDelta,
        altitudeDelta,
        offsetDeltaX,
        offsetDeltaY,
        mouseWheel,
        dt):

        self.azimuth = self.azimuth + 1.0 * dt * -azimuthDelta
        self.altitude = Clamp(self.altitude + 1.0 * dt * altitudeDelta, 0.0, 0.4 * PI)
        self.distance = Clamp(self.distance +  20.0 * dt * -mouseWheel, 0.1, 100.0)
        
        rotationAzimuth = QuaternionFromAxisAngle(Vector3(0, 1, 0), self.azimuth)
        position = Vector3RotateByQuaternion(Vector3(0, 0, self.distance), rotationAzimuth)
        axis = Vector3Normalize(Vector3CrossProduct(position, Vector3(0, 1, 0)))

        rotationAltitude = QuaternionFromAxisAngle(axis, self.altitude)

        localOffset = Vector3(dt * offsetDeltaX, dt * -offsetDeltaY, 0.0)
        localOffset = Vector3RotateByQuaternion(localOffset, rotationAzimuth)

        self.offset = Vector3Add(self.offset, Vector3RotateByQuaternion(localOffset, rotationAltitude))

        cameraTarget = Vector3Add(self.offset, target)
        eye = Vector3Add(cameraTarget, Vector3RotateByQuaternion(position, rotationAltitude))

        self.cam3d.target = cameraTarget
        self.cam3d.position = eye        

#----------------------------------------------------------------------------------
# Shadow Maps
#----------------------------------------------------------------------------------

class ShadowLight:
    
    def __init__(self):
        
        self.target = Vector3Zero()
        self.position = Vector3Zero()
        self.up = Vector3(0.0, 1.0, 0.0)
        self.target = Vector3Zero()
        self.width = 0
        self.height = 0
        self.near = 0.0
        self.far = 1.0


def LoadShadowMap(width, height):

    target = RenderTexture()
    target.id = rlLoadFramebuffer()
    target.texture.width = width
    target.texture.height = height
    assert target.id != 0
    
    rlEnableFramebuffer(target.id)

    target.depth.id = rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19       #DEPTH_COMPONENT_24BIT?
    target.depth.mipmaps = 1
    rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0)
    assert rlFramebufferComplete(target.id)

    rlDisableFramebuffer()

    return target

def UnloadShadowMap(target):
    
    if target.id > 0:
        rlUnloadFramebuffer(target.id)
        

def BeginShadowMap(target, shadowLight):
    
    BeginTextureMode(target)
    ClearBackground(WHITE)
    
    rlDrawRenderBatchActive()      # Update and draw internal render batch

    rlMatrixMode(RL_PROJECTION)    # Switch to projection matrix
    rlPushMatrix()                 # Save previous matrix, which contains the settings for the 2d ortho projection
    rlLoadIdentity()               # Reset current matrix (projection)

    rlOrtho(
        -shadowLight.width/2, shadowLight.width/2, 
        -shadowLight.height/2, shadowLight.height/2, 
        shadowLight.near, shadowLight.far)

    rlMatrixMode(RL_MODELVIEW)     # Switch back to modelview matrix
    rlLoadIdentity()               # Reset current matrix (modelview)

    # Setup Camera view
    matView = MatrixLookAt(shadowLight.position, shadowLight.target, shadowLight.up)
    rlMultMatrixf(MatrixToFloatV(matView).v)      # Multiply modelview matrix by view matrix (camera)

    rlEnableDepthTest()            # Enable DEPTH_TEST for 3D    


def EndShadowMap():
    rlDrawRenderBatchActive()       # Update and draw internal render batch

    rlMatrixMode(RL_PROJECTION)     # Switch to projection matrix
    rlPopMatrix()                   # Restore previous matrix (projection) from matrix stack

    rlMatrixMode(RL_MODELVIEW)      # Switch back to modelview matrix
    rlLoadIdentity()                # Reset current matrix (modelview)

    rlDisableDepthTest()            # Disable DEPTH_TEST for 2D

    EndTextureMode()

def SetShaderValueShadowMap(shader, locIndex, target):
    if locIndex > -1:
        rlEnableShader(shader.id)
        slotPtr = ffi.new('int*'); slotPtr[0] = 10  # Can be anything 0 to 15, but 0 will probably be taken up
        rlActiveTextureSlot(slotPtr[0])
        rlEnableTexture(target.depth.id)
        rlSetUniform(locIndex, slotPtr, SHADER_UNIFORM_INT, 1)

#----------------------------------------------------------------------------------
# GBuffer
#----------------------------------------------------------------------------------

class GBuffer:
    
    def __init__(self):
        self.id = 0              # OpenGL framebuffer object id
        self.color = Texture()   # Color buffer attachment texture 
        self.normal = Texture()  # Normal buffer attachment texture 
        self.depth = Texture()   # Depth buffer attachment texture


def LoadGBuffer(width, height):
    
    target = GBuffer()
    target.id = rlLoadFramebuffer()
    assert target.id
    
    rlEnableFramebuffer(target.id)

    target.color.id = rlLoadTexture(ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, 1)
    target.color.width = width
    target.color.height = height
    target.color.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    target.color.mipmaps = 1
    rlFramebufferAttach(target.id, target.color.id, RL_ATTACHMENT_COLOR_CHANNEL0, RL_ATTACHMENT_TEXTURE2D, 0)
    
    target.normal.id = rlLoadTexture(ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R16G16B16A16, 1)
    target.normal.width = width
    target.normal.height = height
    target.normal.format = PIXELFORMAT_UNCOMPRESSED_R16G16B16A16
    target.normal.mipmaps = 1
    rlFramebufferAttach(target.id, target.normal.id, RL_ATTACHMENT_COLOR_CHANNEL1, RL_ATTACHMENT_TEXTURE2D, 0)
    
    target.depth.id = rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19       #DEPTH_COMPONENT_24BIT?
    target.depth.mipmaps = 1
    rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0)

    assert rlFramebufferComplete(target.id)

    rlDisableFramebuffer()

    return target


def UnloadGBuffer(target):

    if target.id > 0:
        rlUnloadFramebuffer(target.id)


def BeginGBuffer(target, camera):
    
    rlDrawRenderBatchActive()       # Update and draw internal render batch

    rlEnableFramebuffer(target.id)  # Enable render target
    rlActiveDrawBuffers(2) 

    # Set viewport and RLGL internal framebuffer size
    rlViewport(0, 0, target.color.width, target.color.height)
    rlSetFramebufferWidth(target.color.width)
    rlSetFramebufferHeight(target.color.height)

    ClearBackground(BLACK)

    rlMatrixMode(RL_PROJECTION)    # Switch to projection matrix
    rlPushMatrix()                 # Save previous matrix, which contains the settings for the 2d ortho projection
    rlLoadIdentity()               # Reset current matrix (projection)

    aspect = float(target.color.width)/float(target.color.height)

    # NOTE: zNear and zFar values are important when computing depth buffer values
    if camera.projection == CAMERA_PERSPECTIVE:

        # Setup perspective projection
        top = rlGetCullDistanceNear()*np.tan(camera.fovy*0.5*DEG2RAD)
        right = top*aspect

        rlFrustum(-right, right, -top, top, rlGetCullDistanceNear(), rlGetCullDistanceFar())

    elif camera.projection == CAMERA_ORTHOGRAPHIC:

        # Setup orthographic projection
        top = camera.fovy/2.0
        right = top*aspect

        rlOrtho(-right, right, -top,top, rlGetCullDistanceNear(), rlGetCullDistanceFar())

    rlMatrixMode(RL_MODELVIEW)     # Switch back to modelview matrix
    rlLoadIdentity()               # Reset current matrix (modelview)

    # Setup Camera view
    matView = MatrixLookAt(camera.position, camera.target, camera.up)
    rlMultMatrixf(MatrixToFloatV(matView).v)      # Multiply modelview matrix by view matrix (camera)

    rlEnableDepthTest()            # Enable DEPTH_TEST for 3D


def EndGBuffer(windowWidth, windowHeight):
    
    rlDrawRenderBatchActive()       # Update and draw internal render batch
    
    rlDisableDepthTest()            # Disable DEPTH_TEST for 2D
    rlActiveDrawBuffers(1) 
    rlDisableFramebuffer()          # Disable render target (fbo)

    rlMatrixMode(RL_PROJECTION)         # Switch to projection matrix
    rlPopMatrix()                   # Restore previous matrix (projection) from matrix stack
    rlLoadIdentity()                    # Reset current matrix (projection)
    rlOrtho(0, windowWidth, windowHeight, 0, 0.0, 1.0)

    rlMatrixMode(RL_MODELVIEW)          # Switch back to modelview matrix
    rlLoadIdentity()                    # Reset current matrix (modelview)


#----------------------------------------------------------------------------------
# Geno Character and Animation
#----------------------------------------------------------------------------------

def FileRead(out, size, f):
    ffi.memmove(out, f.read(size), size)

def LoadGenoModel(fileName):

    model = Model()
    model.transform = MatrixIdentity()
  
    with open(fileName, "rb") as f:
        
        model.materialCount = 1
        model.materials = MemAlloc(model.materialCount * ffi.sizeof(Mesh()))
        model.materials[0] = LoadMaterialDefault()

        model.meshCount = 1
        model.meshMaterial = MemAlloc(model.meshCount * ffi.sizeof(Mesh()))
        model.meshMaterial[0] = 0

        model.meshes = MemAlloc(model.meshCount * ffi.sizeof(Mesh()))
        model.meshes[0].vertexCount = struct.unpack('I', f.read(4))[0]
        model.meshes[0].triangleCount = struct.unpack('I', f.read(4))[0]
        model.boneCount = struct.unpack('I', f.read(4))[0]

        model.meshes[0].boneCount = model.boneCount
        model.meshes[0].vertices = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.meshes[0].texcoords = MemAlloc(model.meshes[0].vertexCount * 2 * ffi.sizeof("float"))
        model.meshes[0].normals = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.meshes[0].boneIds = MemAlloc(model.meshes[0].vertexCount * 4 * ffi.sizeof("unsigned char"))
        model.meshes[0].boneWeights = MemAlloc(model.meshes[0].vertexCount * 4 * ffi.sizeof("float"))
        model.meshes[0].indices = MemAlloc(model.meshes[0].triangleCount * 3 * ffi.sizeof("unsigned short"))
        model.meshes[0].animVertices = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.meshes[0].animNormals = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.bones =  MemAlloc(model.boneCount * ffi.sizeof(BoneInfo()))
        model.bindPose =  MemAlloc(model.boneCount * ffi.sizeof(Transform()))
        
        FileRead(model.meshes[0].vertices, ffi.sizeof("float") * model.meshes[0].vertexCount * 3, f)
        FileRead(model.meshes[0].texcoords, ffi.sizeof("float") * model.meshes[0].vertexCount * 2, f)
        FileRead(model.meshes[0].normals, ffi.sizeof("float") * model.meshes[0].vertexCount * 3, f)
        FileRead(model.meshes[0].boneIds, ffi.sizeof("unsigned char") * model.meshes[0].vertexCount * 4, f)
        FileRead(model.meshes[0].boneWeights, ffi.sizeof("float") * model.meshes[0].vertexCount * 4, f)
        FileRead(model.meshes[0].indices, ffi.sizeof("unsigned short") * model.meshes[0].triangleCount * 3, f)
        ffi.memmove(model.meshes[0].animVertices, model.meshes[0].vertices, ffi.sizeof("float") * model.meshes[0].vertexCount * 3)
        ffi.memmove(model.meshes[0].animNormals, model.meshes[0].normals, ffi.sizeof("float") * model.meshes[0].vertexCount * 3)
        FileRead(model.bones, ffi.sizeof(BoneInfo()) * model.boneCount, f)
        FileRead(model.bindPose, ffi.sizeof(Transform()) * model.boneCount, f)
        
        model.meshes[0].boneMatrices = MemAlloc(model.boneCount * ffi.sizeof(Matrix()))
        for i in range(model.boneCount):
            model.meshes[0].boneMatrices[i] = MatrixIdentity()
    
    UploadMesh(ffi.addressof(model.meshes[0]), True)
    
    return model


def UpdateModelPoseFromNumpyArrays(model, positions, rotations):
    
    assert model.boneCount == len(positions)
    assert model.boneCount == len(rotations)
    
    for boneId in range(model.boneCount):
        
        bindTransform = model.bindPose[boneId]
        bindMatrix = MatrixMultiply(MatrixMultiply(
            MatrixScale(bindTransform.scale.x, bindTransform.scale.y, bindTransform.scale.z),
            QuaternionToMatrix(bindTransform.rotation)),
            MatrixTranslate(bindTransform.translation.x, bindTransform.translation.y, bindTransform.translation.z))

        targetTranslation = Vector3(positions[boneId,0], positions[boneId,1], positions[boneId,2])
        targetRotation = Vector4(rotations[boneId,1], rotations[boneId,2], rotations[boneId,3], rotations[boneId,0])
        targetMatrix = MatrixMultiply(MatrixMultiply(
            MatrixScale(1.0, 1.0, 1.0),
            QuaternionToMatrix(targetRotation)),
            MatrixTranslate(targetTranslation.x, targetTranslation.y, targetTranslation.z))

        model.meshes[0].boneMatrices[boneId] = MatrixMultiply(MatrixInvert(bindMatrix), targetMatrix);

#----------------------------------------------------------------------------------
# Debug Draw
#----------------------------------------------------------------------------------

def DrawTransform(position, rotation, scale):
    
    rotMatrix = QuaternionToMatrix(Vector4(*rotation))
  
    DrawLine3D(
        Vector3(*position),
        Vector3Add(Vector3(*position), Vector3(scale * rotMatrix.m0, scale * rotMatrix.m1, scale * rotMatrix.m2)),
        RED)
        
    DrawLine3D(
        Vector3(*position),
        Vector3Add(Vector3(*position), Vector3(scale * rotMatrix.m4, scale * rotMatrix.m5, scale * rotMatrix.m6)),
        GREEN)
        
    DrawLine3D(
        Vector3(*position),
        Vector3Add(Vector3(*position), Vector3(scale * rotMatrix.m8, scale * rotMatrix.m9, scale * rotMatrix.m10)),
        BLUE)

def DrawSkeleton(positions, rotations, parents, color):
    
    for i in range(len(positions)):
    
        DrawSphereWires(
            Vector3(*positions[i]),
            0.01,
            4,
            6,
            color)

        DrawTransform(positions[i], rotations[i], 0.1)

        if parents[i] != -1:
        
            DrawLine3D(
                Vector3(*positions[i]),
                Vector3(*positions[parents[i]]),
                color)
    

#----------------------------------------------------------------------------------
# App
#----------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Init Window
    
    screenWidth = 1280
    screenHeight = 720
    
    SetConfigFlags(FLAG_VSYNC_HINT)
    InitWindow(screenWidth, screenHeight, b"GenoViewPython")
    SetTargetFPS(60)

    # Shaders
    
    shadowShader = LoadShader(b"./resources/shadow.vs", b"./resources/shadow.fs")
    shadowShaderLightClipNear = GetShaderLocation(shadowShader, b"lightClipNear")
    shadowShaderLightClipFar = GetShaderLocation(shadowShader, b"lightClipFar")
    
    skinnedShadowShader = LoadShader(b"./resources/skinnedShadow.vs", b"./resources/shadow.fs")
    skinnedShadowShaderLightClipNear = GetShaderLocation(skinnedShadowShader, b"lightClipNear")
    skinnedShadowShaderLightClipFar = GetShaderLocation(skinnedShadowShader, b"lightClipFar")
    
    skinnedBasicShader = LoadShader(b"./resources/skinnedBasic.vs", b"./resources/basic.fs")
    skinnedBasicShaderSpecularity = GetShaderLocation(skinnedBasicShader, b"specularity")
    skinnedBasicShaderGlossiness = GetShaderLocation(skinnedBasicShader, b"glossiness")
    skinnedBasicShaderCamClipNear = GetShaderLocation(skinnedBasicShader, b"camClipNear")
    skinnedBasicShaderCamClipFar = GetShaderLocation(skinnedBasicShader, b"camClipFar")

    basicShader = LoadShader(b"./resources/basic.vs", b"./resources/basic.fs")
    basicShaderSpecularity = GetShaderLocation(basicShader, b"specularity")
    basicShaderGlossiness = GetShaderLocation(basicShader, b"glossiness")
    basicShaderCamClipNear = GetShaderLocation(basicShader, b"camClipNear")
    basicShaderCamClipFar = GetShaderLocation(basicShader, b"camClipFar")
    
    lightingShader = LoadShader(b"./resources/post.vs", b"./resources/lighting.fs")
    lightingShaderGBufferColor = GetShaderLocation(lightingShader, b"gbufferColor")
    lightingShaderGBufferNormal = GetShaderLocation(lightingShader, b"gbufferNormal")
    lightingShaderGBufferDepth = GetShaderLocation(lightingShader, b"gbufferDepth")
    lightingShaderSSAO = GetShaderLocation(lightingShader, b"ssao")
    lightingShaderCamPos = GetShaderLocation(lightingShader, b"camPos")
    lightingShaderCamInvViewProj = GetShaderLocation(lightingShader, b"camInvViewProj")
    lightingShaderLightDir = GetShaderLocation(lightingShader, b"lightDir")
    lightingShaderSunColor = GetShaderLocation(lightingShader, b"sunColor")
    lightingShaderSunStrength = GetShaderLocation(lightingShader, b"sunStrength")
    lightingShaderSkyColor = GetShaderLocation(lightingShader, b"skyColor")
    lightingShaderSkyStrength = GetShaderLocation(lightingShader, b"skyStrength")
    lightingShaderGroundStrength = GetShaderLocation(lightingShader, b"groundStrength")
    lightingShaderAmbientStrength = GetShaderLocation(lightingShader, b"ambientStrength")
    lightingShaderExposure = GetShaderLocation(lightingShader, b"exposure")
    lightingShaderCamClipNear = GetShaderLocation(lightingShader, b"camClipNear")
    lightingShaderCamClipFar = GetShaderLocation(lightingShader, b"camClipFar")
    
    ssaoShader = LoadShader(b"./resources/post.vs", b"./resources/ssao.fs")
    ssaoShaderGBufferNormal = GetShaderLocation(ssaoShader, b"gbufferNormal")
    ssaoShaderGBufferDepth = GetShaderLocation(ssaoShader, b"gbufferDepth")
    ssaoShaderCamView = GetShaderLocation(ssaoShader, b"camView")
    ssaoShaderCamProj = GetShaderLocation(ssaoShader, b"camProj")
    ssaoShaderCamInvProj = GetShaderLocation(ssaoShader, b"camInvProj")
    ssaoShaderCamInvViewProj = GetShaderLocation(ssaoShader, b"camInvViewProj")
    ssaoShaderLightViewProj = GetShaderLocation(ssaoShader, b"lightViewProj")
    ssaoShaderShadowMap = GetShaderLocation(ssaoShader, b"shadowMap")
    ssaoShaderShadowInvResolution = GetShaderLocation(ssaoShader, b"shadowInvResolution")
    ssaoShaderCamClipNear = GetShaderLocation(ssaoShader, b"camClipNear")
    ssaoShaderCamClipFar = GetShaderLocation(ssaoShader, b"camClipFar")
    ssaoShaderLightClipNear = GetShaderLocation(ssaoShader, b"lightClipNear")
    ssaoShaderLightClipFar = GetShaderLocation(ssaoShader, b"lightClipFar")
    ssaoShaderLightDir = GetShaderLocation(ssaoShader, b"lightDir")
    
    blurShader = LoadShader(b"./resources/post.vs", b"./resources/blur.fs")
    blurShaderGBufferNormal = GetShaderLocation(blurShader, b"gbufferNormal")
    blurShaderGBufferDepth = GetShaderLocation(blurShader, b"gbufferDepth")
    blurShaderInputTexture = GetShaderLocation(blurShader, b"inputTexture")
    blurShaderCamInvProj = GetShaderLocation(blurShader, b"camInvProj")
    blurShaderCamClipNear = GetShaderLocation(blurShader, b"camClipNear")
    blurShaderCamClipFar = GetShaderLocation(blurShader, b"camClipFar")
    blurShaderInvTextureResolution = GetShaderLocation(blurShader, b"invTextureResolution")
    blurShaderBlurDirection = GetShaderLocation(blurShader, b"blurDirection")

    fxaaShader = LoadShader(b"./resources/post.vs", b"./resources/fxaa.fs")
    fxaaShaderInputTexture = GetShaderLocation(fxaaShader, b"inputTexture")
    fxaaShaderInvTextureResolution = GetShaderLocation(fxaaShader, b"invTextureResolution")
    
    # Objects
    
    groundMesh = GenMeshPlane(20.0, 20.0, 10, 10)
    groundModel = LoadModelFromMesh(groundMesh)
    groundPosition = Vector3(0.0, -0.01, 0.0)
    
    genoModel = LoadGenoModel(b"./resources/Geno.bin")
    genoPosition = Vector3(0.0, 0.0, 0.0)
    
    # Animation
    
    # bvhData = bvh.load("./resources/ground1_subject1.bvh")
    bvhData = bvh.load("./resources/Geno_bind.bvh")
    
    parents = bvhData['parents']
    localPositions = 0.01 * bvhData['positions'].copy().astype(np.float32)
    localRotations = quat.unroll(quat.from_euler(np.radians(bvhData['rotations']), order=bvhData['order']))
    globalRotations, globalPositions = quat.fk(localRotations, localPositions, parents)
    
    animationFrame = 0
    
    # Camera
    
    camera = Camera()
    
    rlSetClipPlanes(0.01, 50.0)
    
    # Shadows
    
    lightDir = Vector3Normalize(Vector3(0.35, -1.0, -0.35))
    
    shadowLight = ShadowLight()
    shadowLight.target = Vector3Zero()
    shadowLight.position = Vector3Scale(lightDir, -5.0)
    shadowLight.up = Vector3(0.0, 1.0, 0.0)
    shadowLight.width = 5.0
    shadowLight.height = 5.0
    shadowLight.near = 0.01
    shadowLight.far = 10.0
    
    shadowWidth = 1024
    shadowHeight = 1024
    shadowInvResolution = Vector2(1.0 / shadowWidth, 1.0 / shadowHeight)
    shadowMap = LoadShadowMap(shadowWidth, shadowHeight)    
    
    # GBuffer and Render Textures
    
    gbuffer = LoadGBuffer(screenWidth, screenHeight)
    lighted = LoadRenderTexture(screenWidth, screenHeight)
    ssaoFront = LoadRenderTexture(screenWidth, screenHeight)
    ssaoBack = LoadRenderTexture(screenWidth, screenHeight)
    
    # UI
    
    drawBoneTransformsPtr = ffi.new('bool*'); drawBoneTransformsPtr[0] = False
    
    # Go
    
    while not WindowShouldClose():
    
        # Animation
        
        animationFrame = (animationFrame + 1) % len(localPositions)
        UpdateModelPoseFromNumpyArrays(genoModel, globalPositions[animationFrame], globalRotations[animationFrame])

        # Shadow Light Tracks Character
        
        hipPosition = Vector3(*globalPositions[animationFrame][0])
        
        shadowLight.target = Vector3(hipPosition.x, 0.0, hipPosition.z)
        shadowLight.position = Vector3Add(shadowLight.target, Vector3Scale(lightDir, -5.0))

        # Update Camera
        
        camera.update(
            Vector3(hipPosition.x, 0.75, hipPosition.z),
            GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
            GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
            GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
            GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
            GetMouseWheelMove(),
            GetFrameTime())
        
        # Render
        
        rlDisableColorBlend()
        
        BeginDrawing()
        
        # Render Shadow Maps
        
        BeginShadowMap(shadowMap, shadowLight)  
        
        lightViewProj = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection())
        lightClipNear = rlGetCullDistanceNear()
        lightClipFar = rlGetCullDistanceFar()

        lightClipNearPtr = ffi.new("float*"); lightClipNearPtr[0] = lightClipNear
        lightClipFarPtr = ffi.new("float*"); lightClipFarPtr[0] = lightClipFar
        
        SetShaderValue(shadowShader, shadowShaderLightClipNear, lightClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(shadowShader, shadowShaderLightClipFar, lightClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedShadowShader, skinnedShadowShaderLightClipNear, lightClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedShadowShader, skinnedShadowShaderLightClipFar, lightClipFarPtr, SHADER_UNIFORM_FLOAT)
        
        groundModel.materials[0].shader = shadowShader
        DrawModel(groundModel, groundPosition, 1.0, WHITE)
        
        genoModel.materials[0].shader = skinnedShadowShader
        DrawModel(genoModel, genoPosition, 1.0, WHITE)
        
        EndShadowMap()
        
        # Render GBuffer
        
        BeginGBuffer(gbuffer, camera.cam3d)
        
        camView = rlGetMatrixModelview()
        camProj = rlGetMatrixProjection()
        camInvProj = MatrixInvert(camProj)
        camInvViewProj = MatrixInvert(MatrixMultiply(camView, camProj))
        camClipNear = rlGetCullDistanceNear()
        camClipFar = rlGetCullDistanceFar()

        camClipNearPtr = ffi.new("float*"); camClipNearPtr[0] = camClipNear
        camClipFarPtr = ffi.new("float*"); camClipFarPtr[0] = camClipFar

        specularityPtr = ffi.new('float*'); specularityPtr[0] = 0.5
        glossinessPtr = ffi.new('float*'); glossinessPtr[0] = 10.0
        
        SetShaderValue(basicShader, basicShaderSpecularity, specularityPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(basicShader, basicShaderGlossiness, glossinessPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(basicShader, basicShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(basicShader, basicShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderSpecularity, specularityPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderGlossiness, glossinessPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)        
        
        groundModel.materials[0].shader = basicShader
        DrawModel(groundModel, groundPosition, 1.0, Color(190, 190, 190, 255))
        
        genoModel.materials[0].shader = skinnedBasicShader
        DrawModel(genoModel, genoPosition, 1.0, ORANGE)       
        
        EndGBuffer(screenWidth, screenHeight)
        
        # Render SSAO and Shadows
        
        BeginTextureMode(ssaoFront)
        
        BeginShaderMode(ssaoShader)
        
        SetShaderValueTexture(ssaoShader, ssaoShaderGBufferNormal, gbuffer.normal)
        SetShaderValueTexture(ssaoShader, ssaoShaderGBufferDepth, gbuffer.depth)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamView, camView)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamProj, camProj)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamInvProj, camInvProj)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamInvViewProj, camInvViewProj)
        SetShaderValueMatrix(ssaoShader, ssaoShaderLightViewProj, lightViewProj)
        SetShaderValueShadowMap(ssaoShader, ssaoShaderShadowMap, shadowMap)
        SetShaderValue(ssaoShader, ssaoShaderShadowInvResolution, ffi.addressof(shadowInvResolution), SHADER_UNIFORM_VEC2)
        SetShaderValue(ssaoShader, ssaoShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderLightClipNear, lightClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderLightClipFar, lightClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderLightDir, ffi.addressof(lightDir), SHADER_UNIFORM_VEC3)
        
        ClearBackground(WHITE)
        
        DrawTextureRec(
            ssaoFront.texture,
            Rectangle(0, 0, ssaoFront.texture.width, -ssaoFront.texture.height),
            Vector2(0.0, 0.0),
            WHITE)

        EndShaderMode()

        EndTextureMode()
        
        # Blur Horizontal
        
        BeginTextureMode(ssaoBack)
        
        BeginShaderMode(blurShader)
        
        blurDirection = Vector2(1.0, 0.0)
        blurInvTextureResolution = Vector2(1.0 / ssaoFront.texture.width, 1.0 / ssaoFront.texture.height)
        
        SetShaderValueTexture(blurShader, blurShaderGBufferNormal, gbuffer.normal)
        SetShaderValueTexture(blurShader, blurShaderGBufferDepth, gbuffer.depth)
        SetShaderValueTexture(blurShader, blurShaderInputTexture, ssaoFront.texture)
        SetShaderValueMatrix(blurShader, blurShaderCamInvProj, camInvProj)
        SetShaderValue(blurShader, blurShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(blurShader, blurShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(blurShader, blurShaderInvTextureResolution, ffi.addressof(blurInvTextureResolution), SHADER_UNIFORM_VEC2)
        SetShaderValue(blurShader, blurShaderBlurDirection, ffi.addressof(blurDirection), SHADER_UNIFORM_VEC2)

        DrawTextureRec(
            ssaoBack.texture,
            Rectangle(0, 0, ssaoBack.texture.width, -ssaoBack.texture.height),
            Vector2(0, 0),
            WHITE)

        EndShaderMode()

        EndTextureMode()
      
        # Blur Vertical
        
        BeginTextureMode(ssaoFront)
        
        BeginShaderMode(blurShader)
        
        blurDirection = Vector2(0.0, 1.0)
        
        SetShaderValueTexture(blurShader, blurShaderInputTexture, ssaoBack.texture)
        SetShaderValue(blurShader, blurShaderBlurDirection, ffi.addressof(blurDirection), SHADER_UNIFORM_VEC2)

        DrawTextureRec(
            ssaoFront.texture,
            Rectangle(0, 0, ssaoFront.texture.width, -ssaoFront.texture.height),
            Vector2(0, 0),
            WHITE)

        EndShaderMode()

        EndTextureMode()
      
        # Light GBuffer
        
        BeginTextureMode(lighted)
        
        BeginShaderMode(lightingShader)
        
        sunColor = Vector3(253.0 / 255.0, 255.0 / 255.0, 232.0 / 255.0)
        sunStrengthPtr = ffi.new('float*'); sunStrengthPtr[0] = 0.25
        skyColor = Vector3(174.0 / 255.0, 183.0 / 255.0, 190.0 / 255.0)
        skyStrengthPtr = ffi.new('float*'); skyStrengthPtr[0] = 0.15
        groundStrengthPtr = ffi.new('float*'); groundStrengthPtr[0] = 0.1
        ambientStrengthPtr = ffi.new('float*'); ambientStrengthPtr[0] = 1.0
        exposurePtr = ffi.new('float*'); exposurePtr[0] = 0.9
        
        SetShaderValueTexture(lightingShader, lightingShaderGBufferColor, gbuffer.color)
        SetShaderValueTexture(lightingShader, lightingShaderGBufferNormal, gbuffer.normal)
        SetShaderValueTexture(lightingShader, lightingShaderGBufferDepth, gbuffer.depth)
        SetShaderValueTexture(lightingShader, lightingShaderSSAO, ssaoFront.texture)
        SetShaderValue(lightingShader, lightingShaderCamPos, ffi.addressof(camera.cam3d.position), SHADER_UNIFORM_VEC3)
        SetShaderValueMatrix(lightingShader, lightingShaderCamInvViewProj, camInvViewProj)
        SetShaderValue(lightingShader, lightingShaderLightDir, ffi.addressof(lightDir), SHADER_UNIFORM_VEC3)
        SetShaderValue(lightingShader, lightingShaderSunColor, ffi.addressof(sunColor), SHADER_UNIFORM_VEC3)
        SetShaderValue(lightingShader, lightingShaderSunStrength, sunStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderSkyColor, ffi.addressof(skyColor), SHADER_UNIFORM_VEC3)
        SetShaderValue(lightingShader, lightingShaderSkyStrength, skyStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderGroundStrength, groundStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderAmbientStrength, ambientStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderExposure, exposurePtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        
        ClearBackground(RAYWHITE)
        
        DrawTextureRec(
            gbuffer.color,
            Rectangle(0, 0, gbuffer.color.width, -gbuffer.color.height),
            Vector2(0, 0),
            WHITE)
        
        EndShaderMode()        
        
        # Debug Draw
        
        BeginMode3D(camera.cam3d)
        
        if drawBoneTransformsPtr[0]:
            DrawSkeleton(
                globalPositions[animationFrame], 
                globalRotations[animationFrame], 
                parents, GRAY)
  
        EndMode3D()

        EndTextureMode()
        
        # Render Final with FXAA
        
        BeginShaderMode(fxaaShader)

        fxaaInvTextureResolution = Vector2(1.0 / lighted.texture.width, 1.0 / lighted.texture.height)
        
        SetShaderValueTexture(fxaaShader, fxaaShaderInputTexture, lighted.texture)
        SetShaderValue(fxaaShader, fxaaShaderInvTextureResolution, ffi.addressof(fxaaInvTextureResolution), SHADER_UNIFORM_VEC2)
        
        DrawTextureRec(
            lighted.texture,
            Rectangle(0, 0, lighted.texture.width, -lighted.texture.height),
            Vector2(0, 0),
            WHITE)
        
        EndShaderMode()
  
        # UI
  
        rlEnableColorBlend()
  
        GuiGroupBox(Rectangle(20, 10, 190, 180), b"Camera")

        GuiLabel(Rectangle(30, 20, 150, 20), b"Ctrl + Left Click - Rotate")
        GuiLabel(Rectangle(30, 40, 150, 20), b"Ctrl + Right Click - Pan")
        GuiLabel(Rectangle(30, 60, 150, 20), b"Mouse Scroll - Zoom")
        GuiLabel(Rectangle(30, 80, 150, 20), b"Target: [% 5.3f % 5.3f % 5.3f]" % (camera.cam3d.target.x, camera.cam3d.target.y, camera.cam3d.target.z))
        GuiLabel(Rectangle(30, 100, 150, 20), b"Offset: [% 5.3f % 5.3f % 5.3f]" % (camera.offset.x, camera.offset.y, camera.offset.z))
        GuiLabel(Rectangle(30, 120, 150, 20), b"Azimuth: %5.3f" % camera.azimuth)
        GuiLabel(Rectangle(30, 140, 150, 20), b"Altitude: %5.3f" % camera.altitude)
        GuiLabel(Rectangle(30, 160, 150, 20), b"Distance: %5.3f" % camera.distance)
  
        GuiGroupBox(Rectangle(screenWidth - 260, 10, 240, 40), b"Rendering")

        GuiCheckBox(Rectangle(screenWidth - 250, 20, 20, 20), b"Draw Transforms", drawBoneTransformsPtr)

  
        EndDrawing()

    UnloadRenderTexture(lighted)
    UnloadRenderTexture(ssaoBack)
    UnloadRenderTexture(ssaoFront)
    UnloadRenderTexture(lighted)
    UnloadGBuffer(gbuffer)

    UnloadShadowMap(shadowMap)
    
    UnloadModel(genoModel)
    UnloadModel(groundModel)
    
    UnloadShader(fxaaShader)    
    UnloadShader(blurShader)    
    UnloadShader(ssaoShader) 
    UnloadShader(lightingShader)    
    UnloadShader(basicShader)
    UnloadShader(skinnedBasicShader)
    UnloadShader(skinnedShadowShader)
    UnloadShader(shadowShader)
    
    CloseWindow()