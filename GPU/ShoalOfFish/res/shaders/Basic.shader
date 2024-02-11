#shader vertex
#version 330 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 in_color;

out vec3 out_color;

void main()
{
        gl_Position = vec4(position.x, position.y, 0.0, 1.0);
        out_color = in_color;
};

#shader fragment
#version 330 core

out vec4 color;
in vec3 out_color;
void main()
{
    color = vec4(out_color, 1.0);
};