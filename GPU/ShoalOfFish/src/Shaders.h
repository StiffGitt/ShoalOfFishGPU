#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <GL/glew.h>


unsigned int StartShaders(const std::string& filepath);
unsigned int CompileShader(unsigned int type, const std::string& source);
unsigned int CreateShader(const std::string& vertexShader, const std::string& fragmentShader);