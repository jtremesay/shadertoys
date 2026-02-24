from py2glsl import ShaderContext
from py2glsl.builtins import sin, vec4
from py2glsl.render import animate


def shader(ctx: ShaderContext) -> vec4:
    return vec4(ctx.vs_uv, sin(ctx.u_time) / 2 + 0.5, 1.0)


if __name__ == "__main__":
    animate(shader, fps=30)
