#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <cstdlib> 
#include <ctime> 
#include <Windows.h>
#include <algorithm>
#include <iterator>
#include "Shaders.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "Fish.h"
#include "kernel.cuh"
#include "Consts.h"


GLFWwindow* window;
GLuint buffer, vao;

float A[3] = { 0.005f, 0.007f, 0.01f };
float H[3] = { 0.02f, 0.03f, 0.04f };
double cursorX = 0;
double cursorY = 0;
bool mouse_pressed = false;
int N;

int window_init();
Fish* fishes_init();
float* get_vertices(Fish* fishes);
void buffer_init(Fish* fishes);
void draw_frame(float* vertices);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);

int main(int argc, char *argv[])
{
    // Initialize fishes count
    if (argc > 1)
        N = std::stoi(argv[1]);
    else
        N = 1000;

    // Initialize model coefficients
    float r1 = RANGE1, cohensionCoef = 0.25, avoidCoef = 0.5, alignCoef = 0.5, predatorsCoef = 0.5f, preyCoef = 0.3f, turnCoef = TURN_COEF;
    float* vertices = (float*)malloc(N * 3 * ATTR_COUNT * sizeof(float));

    if (window_init())
        return -1;

    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    Fish *fishes = fishes_init();

    // Grid parameters
    float cell_size = RANGE2;
    int grid_length = ((int)(2.0f / cell_size) + 1) * ((int)(2.0f / cell_size) + 1);

    init_cuda(N, grid_length, fishes);

    buffer_init(fishes);

    GLuint shader = StartShaders("res/shaders/Basic.shader");
     
    // Initialize Imgui
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    bool shouldPause = false, predatorMode = false;
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        if (!shouldPause)
        {
            make_calculations_cuda(vertices, r1, RANGE2, turnCoef, cohensionCoef / 1000.0f, avoidCoef / 100.0f, alignCoef / 100.0f, predatorsCoef / 50.0f,
                preyCoef / 100.0f, MAXV, MINV, ((cursorX / WINDOW_WIDTH) * 2) - 1, ((cursorY / WINDOW_HEIGHT) * 2) - 1, mouse_pressed,
                predatorMode);
            draw_frame(vertices);
        }


        // Imgui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Begin("Configuration");
        ImGui::SliderFloat("Separation range", &r1, 0.0f, RANGE2);
        ImGui::SliderFloat("Separation", &avoidCoef, 0.0f, 1.0f);
        ImGui::SliderFloat("Cohension", &cohensionCoef, 0.0f, 1.0f);
        ImGui::SliderFloat("Alignment", &alignCoef, 0.0f, 1.0f);
        ImGui::Checkbox("Predator mode", &predatorMode);
        ImGui::SliderFloat("Avoid predators", &predatorsCoef, 0.0f, 1.0f);
        ImGui::SliderFloat("Chase prey", &preyCoef, 0.0f, 1.0f);
        if (ImGui::Button("Pause"))
            shouldPause = !shouldPause;

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap front and back buffers 
        glfwSwapBuffers(window);
    }

    // Free resources
    glDeleteProgram(shader);
    free_cuda();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}

int window_init()
{
    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Shoal Of Fish", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK)
        std::cout << "glewInit ERROR" << std::endl;

    std::cout << glGetString(GL_VERSION) << std::endl;

    return 0;
}

// Initialize random fishes position and velocity
Fish* fishes_init()
{
    Fish* fishes = (Fish*)malloc(N * sizeof(Fish));
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (int i = 0; i < N; i++)
    {
        fishes[i].x = (static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX)) * 2.0f - 1.0f;
        fishes[i].y = (static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX)) * 2.0f - 1.0f;
        fishes[i].dx = (static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX) / 50.0f) * ( (std::rand() % 2)? -1.0f : 1.0f);
        fishes[i].dy = (static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX) / 50.0f) * ( (std::rand() % 2) ? -1.0f : 1.0f);
        if (i % 50 == 0)
            fishes[i].species = 2;
        else if (i % 10 == 0)
            fishes[i].species = 1;
        else
            fishes[i].species = 0;
    }
    return fishes;
}

// Calculate triangle vertices position for each fish
float* get_vertices(Fish *fishes)
{
    float* vertices = (float*)malloc(N * 3 * ATTR_COUNT * sizeof(float));
    for (int i = 0; i < N; i++)
    {
        float x = fishes[i].x;
        float y = fishes[i].y;
        float dx = fishes[i].dx;
        float dy = fishes[i].dy;
        int species = fishes[i].species;
        float d = sqrtf(dx * dx + dy * dy);

        vertices[i * 3 * ATTR_COUNT] = x - A[species] * (dy / d);
        vertices[i * 3 * ATTR_COUNT + 1] = y + A[species] * (dx / d);
        vertices[i * 3 * ATTR_COUNT + 2] = GET_R(species);
        vertices[i * 3 * ATTR_COUNT + 3] = GET_G(species);
        vertices[i * 3 * ATTR_COUNT + 4] = GET_B(species);

        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT] = x + A[species] * (dy / d);
        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT + 1] = y - A[species] * (dx / d);
        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT + 2] = GET_R(species);
        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT + 3] = GET_G(species);
        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT + 4] = GET_B(species);

        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT * 2] = x + H[species] * (dx / d);
        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT * 2 + 1] = y + H[species] * (dy / d);
        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT * 2 + 2] = GET_R(species);
        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT * 2 + 3] = GET_G(species);
        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT * 2 + 4] = GET_B(species);
    }
    return vertices;
}

// Initialize vertices buffer and vertex array
void buffer_init(Fish* fishes)
{
    float* vertices = get_vertices(fishes);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, 3 * ATTR_COUNT * N * sizeof(float), vertices, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, ATTR_COUNT * sizeof(float), (void*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, ATTR_COUNT * sizeof(float), (void*)(sizeof(float) * 2));

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void draw_frame(float *vertices)
{
    glClear(GL_COLOR_BUFFER_BIT);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 3 * ATTR_COUNT * N * sizeof(float), vertices);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glDrawArrays(GL_TRIANGLES, 0, 3 * N);

    glBindVertexArray(0);
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (mouse_pressed)
    {
        cursorX = xpos;
        cursorY = WINDOW_HEIGHT - ypos;
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        cursorX = xpos;
        cursorY = WINDOW_HEIGHT - ypos;
        mouse_pressed = true;
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
    {
        mouse_pressed = false;
    }
}