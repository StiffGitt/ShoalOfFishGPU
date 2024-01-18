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

struct Velocity {
    float x[N];
    float y[N];
};

class sort_indices
{
private:
    int* mparr;
public:
    sort_indices(int* parr) : mparr(parr) {}
    bool operator()(int i, int j) const { return mparr[i] < mparr[j]; }
};

GLFWwindow* window;
GLuint buffer, vao;
Fish fishes[N];
Fish gathered_fishes[N];

float A[3] = { 0.005f, 0.007f, 0.01f };
float H[3] = { 0.02f, 0.03f, 0.04f };
double cursorX = 0;
double cursorY = 0;
bool mouse_pressed = false;

int window_init();
void fishes_init();
float* get_vertices();
void buffer_init();
void draw_frame();
void move_fishes();
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);


void print_vertices(float* vertices);
void print_fish();


int main(void)
{
    float r1 = RANGE1, cohensionCoef = 0.25, avoidCoef = 0.5, alignCoef = 0.5, predatorsCoef = 0.5f, preyCoef = 0.3f;
    float turnCoef = TURN_COEF;
    if (window_init())
        return -1;

    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    fishes_init();
    float cell_size = RANGE2 * 2;
    int grid_length = ((int)(2.0f / cell_size) + 1) * ((int)(2.0f / cell_size) + 1);
    init_cuda(grid_length);
    buffer_init();

    GLuint shader = StartShaders("res/shaders/Basic.shader");
     
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

        draw_frame();

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

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        //Sleep(20);
        if (!shouldPause)
        {
            make_calculations_cuda(fishes, r1, RANGE2, turnCoef, cohensionCoef / 1000.0f, avoidCoef / 100.0f, alignCoef / 100.0f, predatorsCoef / 50.0f,
                preyCoef / 100.0f, MAXV, MINV, ((cursorX / WINDOW_WIDTH) * 2) - 1, ((cursorY / WINDOW_HEIGHT) * 2) - 1, mouse_pressed,
                predatorMode);
            /*calculate_v(r1, RANGE2, turnCoef, cohensionCoef / 1000.0f, avoidCoef / 100.0f, alignCoef / 100.0f, predatorsCoef / 50.0f,
                preyCoef / 100.0f, MAXV, MINV, ((cursorX / WINDOW_WIDTH) * 2) - 1, ((cursorY / WINDOW_HEIGHT) * 2) - 1, mouse_pressed, 
                predatorMode);*/

            //move_fishes();
        }
    }

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

void fishes_init()
{
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
}

float* get_vertices()
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

void buffer_init()
{
    float* vertices = get_vertices();

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

void draw_frame()
{
    float* vertices = get_vertices();
    //print_vertices(vertices);
    /* Render here */
    glClear(GL_COLOR_BUFFER_BIT);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 3 * ATTR_COUNT * N * sizeof(float), vertices);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glDrawArrays(GL_TRIANGLES, 0, 3 * N);

    glBindVertexArray(0);
}

void move_fishes()
{
    for (int i = 0; i < N; i++)
    {
        fishes[i].x += fishes[i].dx;
        fishes[i].y += fishes[i].dy;
    }
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

void print_vertices(float* vertices)
{
    for (int i = 0; i < 3 * N; i++)
    {
        //if (vertices[i * ATTR_COUNT] > 1.0f)
        //    vertices[i * ATTR_COUNT] = 1.0f;
        //if (vertices[i * ATTR_COUNT] < 0.0f)
        //    vertices[i * ATTR_COUNT] = 0.0f;
        //if (vertices[i * ATTR_COUNT + 1] > 1.0f)
        //    vertices[i * ATTR_COUNT + 1] = 1.0f;
        //if (vertices[i * ATTR_COUNT + 1] < 0.0f)
        //    vertices[i * ATTR_COUNT + 1] = 0.0f;
        std::cout << "x1 = " << vertices[i  * ATTR_COUNT] << std::endl;
        std::cout << "y1 = " << vertices[i  * ATTR_COUNT + 1] << std::endl;
        std::cout << "r = " << vertices[i * ATTR_COUNT + 2] << std::endl;
        std::cout << "g = " << vertices[i * ATTR_COUNT + 3] << std::endl;
        std::cout << "b = " << vertices[i * ATTR_COUNT + 4] << std::endl;

        std::cout << "---------------------" << std::endl;
    }

    /*for (int i = 0; i < 3 * N; i++)
    {
        std::cout << vertices[i * ATTR_COUNT] << " , ";
        std::cout << vertices[i * ATTR_COUNT + 1] << " , ";
        std::cout << vertices[i * ATTR_COUNT + 2] << " , ";
        std::cout << vertices[i * ATTR_COUNT + 3] << " , ";
        std::cout << vertices[i * ATTR_COUNT + 4] << " , " << std::endl;
    }*/
}

void print_fish()
{
    for (int i = 0; i < N; i++)
    {
        std::cout << "vx = " << fishes[i].dx << std::endl;
        std::cout << "vy = " << fishes[i].dy << std::endl;
    }
}