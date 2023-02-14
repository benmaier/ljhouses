"""
Visualizations of simulations with pyglet.
"""

from sys import platform

from copy import deepcopy
from itertools import chain

import numpy as np

import pyglet
from pyglet import shapes
from pyglet.window import key, mouse, Window
from pyglet.gl import *

from ljhouses import (
        simulate_once,
        _total_interaction_energy,
        _total_kinetic_energy,
        _total_potential_energy,
    )

from ljhouses.pythonsims import simulate_once_python

_colors = [
            '#00997D', # paolo veronese green
            '#a20252', # Jazzberry Jam
            '#cc5803', # Tenne Tawny
            '#81d473', # Mantis
            '#745296', # Royal purple
            '#555555', #grey
            '#f5cc00', # Jonquil 
        ]


#_colors = list(colors.values())


if platform == "linux" or platform == "linux2":
    _IS_LINUX = True
elif platform == "darwin":
    _IS_LINUX = False
elif platform == "win32":
    _IS_LINUX = False


class SimulationStatus():
    """
    Saves information about the current simulation.

    Parameters
    ==========
    N : int
        Number of nodes
    N_steps_per_frame : float
        The amount of simulation time that's supposed
        to pass during a single update

    Attributes
    ==========
    old_node_status : numpy.ndarray
        An array containing node statuses of the previous update
    N_steps_per_frame : float
        The amount of simulation time that's supposed
        to pass during a single update
    simulation_ended : bool
        Whether or not the simulation is over
    paused : bool
        Whether or not the simulation is paused
    """

    def __init__(self,N,N_steps_per_frame):
        self.old_node_status = -1e300*np.ones((N,))
        self.simulation_ended = False
        self.N_steps_per_frame = N_steps_per_frame
        self.paused = False

    def update(self,x,v,a):
        """
        Update the nodes statuses.
        """
        self.x = np.array(x)
        self.v = np.array(v)
        self.a = np.array(a)

    def set_simulation_status(self,simulation_ended):
        """
        Trigger the simulation to be over.
        """
        self.simulation_ended = simulation_ended


class App(pyglet.window.Window):
    """

    A pyglet Window class that makes zooming and panning convenient
    and tracks user input.

    Adapted from Peter Varo's solution
    at https://stackoverflow.com/a/19453006/4177832

    Parameters
    ==========
    width : float
        Width of the app window
    height : float
        Height of the app window
    simulation_status : SimulationStatus
        An object that tracks the simulation. Here,
        it's used to pause or increase the simulation speed.
    """

    def __init__(self, width, height, simulation_status, *args, **kwargs):
        #conf = Config(sample_buffers=1,
        #              samples=4,
        #              depth_size=16,
        #              double_buffer=True)
        self.left   = 0
        self.right  = width
        self.bottom = 0
        self.top    = height
        super().__init__(width, height, *args, **kwargs)
        self.batches = []
        self.batch_funcs = []

        #Initialize camera values
        self.left   = 0
        self.right  = width
        self.bottom = 0
        self.top    = height
        self.zoom_level = 1
        self.zoomed_width  = width
        self.zoomed_height = height

        # Set window values
        self.width  = width
        self.height = height

        self.orig_left = self.left
        self.orig_right = self.right
        self.orig_bottom = self.bottom
        self.orig_top = self.top
        self.orig_zoom_level = self.zoom_level
        self.orig_zoomed_width = self.zoomed_width
        self.orig_zoomed_height = self.zoomed_height

        self.simulation_status = simulation_status

        self.has_been_resized = False


    def add_batch(self,batch,prefunc=None):
        """
        Add a batch that needs to be drawn.
        Optionally, also pass a function that's
        triggered before this batch is drawn.
        """
        self.batches.append(batch)
        self.batch_funcs.append(prefunc)


    def on_draw(self):
        """
        Clear and draw all batches
        """
        self.clear()

        for batch, func in zip(self.batches, self.batch_funcs):
            if func is not None:
                func()
            batch.draw()

    def init_gl(self, width, height):
        # Set viewport
        glMatrixMode( GL_PROJECTION )
        glLoadIdentity()
        #try:
        glOrtho( self.left, self.right, self.bottom, self.top, 1, -1 )
        #except AttributeError as e:
        #    print(self.name) 

    def on_resize(self, width, height):
        """Rescale."""
        # Set window values
        #self.width  = width
        #self.height = height

        # Initialize OpenGL context
        if (_IS_LINUX and not self.has_been_resized) or not _IS_LINUX:
            self.init_gl(width, height)

        self.has_been_resized = True

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Pan."""
        # Move camera
        self.left   -= dx*self.zoom_level
        self.right  -= dx*self.zoom_level
        self.bottom -= dy*self.zoom_level
        self.top    -= dy*self.zoom_level

        glMatrixMode( GL_PROJECTION )
        glLoadIdentity()
        glOrtho( self.left, self.right, self.bottom, self.top, 1, -1 )

    def on_mouse_scroll(self, x, y, dx, dy):
        """Zoom."""
        # Get scale factor
        f = 1+dy/50
        # If zoom_level is in the proper range
        if .02 < self.zoom_level*f < 10:

            self.zoom_level *= f

            mouse_x = x/self.width
            mouse_y = y/self.height

            mouse_x_in_world = self.left   + mouse_x*self.zoomed_width
            mouse_y_in_world = self.bottom + mouse_y*self.zoomed_height

            self.zoomed_width  *= f
            self.zoomed_height *= f

            self.left   = mouse_x_in_world - mouse_x*self.zoomed_width
            self.right  = mouse_x_in_world + (1 - mouse_x)*self.zoomed_width
            self.bottom = mouse_y_in_world - mouse_y*self.zoomed_height
            self.top    = mouse_y_in_world + (1 - mouse_y)*self.zoomed_height

        glMatrixMode( GL_PROJECTION )
        glLoadIdentity()
        glOrtho( self.left, self.right, self.bottom, self.top, 1, -1 )


    def on_key_press(self, symbol, modifiers):
        """
        Check for keyboard input.
        Current inputs:

        - backspace or CMD+0: reset view
        - up : increase simulation speed
        - down : decrease simulation speed
        - space : pause simulation
        """
        #if symbol & key.BACKSPACE or (symbol & key._0 and (modifiers & MOD_COMMAND or modifiers & MOD_CTRL)):
        if symbol == key.BACKSPACE or (symbol == key._0 and (modifiers & key.MOD_COMMAND)):
            self.left = self.orig_left
            self.right = self.orig_right
            self.bottom = self.orig_bottom
            self.top = self.orig_top
            self.zoom_level = self.orig_zoom_level
            self.zoomed_width = self.orig_zoomed_width
            self.zoomed_height = self.orig_zoomed_height

            glMatrixMode( GL_PROJECTION )
            glLoadIdentity()
            glOrtho( self.left, self.right, self.bottom, self.top, 1, -1 )
        elif symbol == key.UP:
            self.simulation_status.N_steps_per_frame *= 1.2
            self.simulation_status.N_steps_per_frame = int(self.simulation_status.N_steps_per_frame)
        elif symbol == key.DOWN:
            self.simulation_status.N_steps_per_frame /= 1.2
            self.simulation_status.N_steps_per_frame = int(self.simulation_status.N_steps_per_frame)
        elif symbol == key.SPACE:
            self.simulation_status.paused = not self.simulation_status.paused



class Scale():
    """
    A scale that maps all its connected graphics objects
    to world (window) dimensions.

    Parameters
    ==========
    bound_increase_factor : float, default = 1.0
        By how much the respective bound should increase
        once it's reached.

    Attributes
    ==========
    bound_increase_factor : float
        By how much the respective bound should increase
        once it's reached.
    x0 : float
        lower bound of data x-dimension
    x1 : float
        upper bound of data x-dimension
    y0 : float
        lower bound of data y-dimension
    y1 : float
        upper bound of data y-dimension
    left : float
        lower bound of world x-dimension
    right : float
        upper bound of world x-dimension
    bottom : float
        lower bound of world y-dimension
    top : float
        upper bound of world y-dimension
    scaling_objects : list
        A list of objects that need to be rescaled
        once the data or world dimensions change.
        Each entry of this list is assumed to be 
        an object that has a method called ``rescale()``.
    """

    def __init__(self,bound_increase_factor=1.0,is_square_domain=False,enforce_center=None):

        self.x0 = np.nan
        self.y0 = np.nan
        self.x1 = np.nan
        self.y1 = np.nan
        self.left = np.nan
        self.right = np.nan
        self.bottom = np.nan
        self.top = np.nan
        self.is_square_domain = is_square_domain
        self.enforce_center = enforce_center

        self.bound_increase_factor = bound_increase_factor

        self.scaling_objects = []

    def extent(self,left,right,top,bottom):
        """
        Define the world (window) dimensions.
        """
        if self.is_square_domain:
            if right-left != top-bottom:
                raise ValueError('Domain was set to be square but these extent values are not')
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self._calc()
        return self

    def domain(self,x0,x1,y0,y1):
        """
        Define the data dimensions.
        """
        if self.is_square_domain:
            if x1-x0 != y1-y0:
                raise ValueError('Domain was set to be square but these domain values are not')
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self._calc()
        return self

    def _calc(self):
        """
        calculate scalars
        """
        self.mx = (self.right-self.left)/(self.x1 - self.x0)
        self.my = (self.top-self.bottom)/(self.y1 - self.y0)

    def scale(self,x,y):
        """
        Scale data.
        """
        _x = self.scalex(x)
        _y = self.scaley(y)

        return _x, _y

    def scalex(self,x):
        """
        Scale x-data
        """
        if type(x) == list:
            _x = list(map(lambda _x: self.mx * (_x-self.x0) + self.left, x ))
        else:
            _x = self.mx * (x-self.x0) + self.left

        return _x

    def scaley(self,y):
        """
        Scale y-data
        """
        if type(y) == list:
            _y = list(map(lambda _y: self.my * (_y-self.y0) + self.bottom, y ))
        else:
            _y = self.my * (y-self.y0) + self.bottom

        return _y

    def scale_dist(self,r):
        if not self.is_square_domain:
            ValueError('this only works for square domains')
        if hasattr(r,'__len__'):
            _r = list(map(lambda _r: self.my * _r, r ))
        else:
            _r = self.my * r

        return _r

    def check_bounds(self,xmin,xmax,ymin,ymax):
        """
        Check whether the global data dimensions have changed
        considered updated data dimensions of a single instance.
        If this is the case, trigger rescaling of all connected
        instances.
        """
        changed = False

        x0, x1, y0, y1 = self.x0, self.x1, self.y0, self.y1
        if xmin < self.x0:
            xvec = xmin - self.x1
            x0 = self.x1 + xvec * self.bound_increase_factor
            changed = True
        if ymin < self.y0:
            yvec = ymin - self.y1
            y0 = self.y1 + yvec * self.bound_increase_factor
            changed = True
        if xmax > self.x1:
            xvec = xmax - self.x0
            x1 = self.x0 + xvec * self.bound_increase_factor
            changed = True
        if ymax > self.y1:
            yvec = ymax - self.y0
            y1 = self.y0 + yvec * self.bound_increase_factor
            changed = True

        if changed:
            self.domain(x0,x1,y0,y1)
            for obj in self.scaling_objects:
                obj.rescale()

    def check_square_bounds_centered_on_origin(self,xdata,ydata,origin=[0.,0.]):

        if not self.is_square_domain:
            raise ValueError('this only works on forced square domains')

        changed = False
        ox = origin[0]
        oy = origin[1]
        maxx = np.max(np.abs(xdata-ox))
        maxy = np.max(np.abs(ydata-oy))
        amax = max(maxx, maxy)
        v = max((self.x1 - ox), (ox-self.x0))
        bound_decrease_factor = 1/self.bound_increase_factor
        if amax < v * bound_decrease_factor:
            v *= bound_decrease_factor
            x0 = ox - v
            y0 = oy - v
            x1 = ox + v
            y1 = oy + v
            changed = True
        elif amax > v:
            v *= self.bound_increase_factor
            x0 = ox - v
            y0 = oy - v
            x1 = ox + v
            y1 = oy + v
            changed = True

        if changed:
            self.domain(x0,x1,y0,y1)
            for obj in self.scaling_objects:
                obj.rescale()


    def add_scaling_object(self,obj):
        """
        Append an object that depends on this Scale instance.
        """
        self.scaling_objects.append(obj)



class Curve():
    """
    A class that draws an OpenGL
    curve to a pyglet Batch instance
    with easy methods to update data.

    Parameters
    ==========
        x : list
            x data
        y : list
            y data
        color : list 
            List of 3 integers between 0 and 255 (RGB color list)
        scale : Scale
            An instance of a Scale that maps the data dimensions
            to an area in a pyglet Window

    Attributes
    ==========
        batch : pyglet.graphics.Batch
            The batch instance in which this curve is drawn
        scale : Scale
            An instance of a Scale that maps the data dimensions
            to an area in a pyglet Window
        vertex_list : pyglet.graphics.VertexList
            Contains the vertex list in window coordinates.
            format strings are ``v2f`` and ``c3B``.
        color : list
            as described above
        xmin : float
            lower bound of x-dimension
        xmax : float
            upper bound of x-dimension
        ymin : float
            lower bound of y-dimension
        ymax : float
            upper bound of y-dimension
    """

    def __init__(self,x,y,color,scale,batch):
        self.batch = batch
        self.vertex_list = batch.add(0, GL_LINE_STRIP, None,
                    'v2f',
                    'c3B',
                )
        self.scale = scale
        scale.add_scaling_object(self)
        self.color = color
        self.xmin = 1e300
        self.ymin = 1e300
        self.xmax = -1e300
        self.ymax = -1e300
        self.set(x,y)

    def set(self,x,y):
        """
        Set the data of this curve.
        """
        _x = list(x)
        _y = list(y)

        # GL_LINE_STRIP needs to have the first
        # and last vertex as duplicates,
        # save data
        self.x = _x[:1] + _x + _x[-1:]
        self.y = _y[:1] + _y + _y[-1:]

        # get min/max values of this update
        xmin = min(_x)
        xmax = max(_x)
        ymin = min(_y)
        ymax = max(_y)

        # scale and zip together the new numbers
        _x, _y = self.scale.scale(self.x, self.y)
        xy = list(chain.from_iterable(zip(_x, _y)))

        # resize vertex list, set vertices and colors
        self.vertex_list.resize(len(_x))
        self.vertex_list.vertices = xy
        self.vertex_list.colors = self.color * len(_x)

        # check whether or not the bounds of
        # the scale need to be updated
        self.update_bounds(xmin,xmax,ymin,ymax)

    def append_single_value(self,x,y):
        """
        Append a single data point to this curve.
        Note that if the bounds change with this
        update, the connect Scale-instance will be updated
        automatically.
        """
        self.append_list([x], [y])

    def append_list(self,x,y):
        """
        Append a list of data points to this curve.
        Note that if the bounds change with this
        update, the connect Scale-instance will be updated
        automatically.
        """
        _x = x + x[-1:]
        _y = y + y[-1:]

        # remember that self.x contains the last
        # vertex twice for GL_LINE_STRIP.
        # We have to pop the duplicate of the
        # formerly last entry and append the new list
        self.x.pop()
        self.x.extend(_x)
        self.y.pop()
        self.y.extend(_y)

        xmin = min(_x)
        xmax = max(_x)
        ymin = min(_y)
        ymax = max(_y)

        _x, _y = self.scale.scale(_x, _y)
        xy = list(chain.from_iterable(zip(_x, _y)))

        self.vertex_list.resize(self.vertex_list.get_size() + len(_x) -1 )
        self.vertex_list.vertices[-len(xy):] = xy
        self.vertex_list.colors[-3*len(_x):] = self.color * len(_x)

        self.update_bounds(xmin,xmax,ymin,ymax)

    def rescale(self):
        """
        Rescale this curve's data according to
        the connected Scale-instance
        """
        _x, _y = self.scale.scale(self.x, self.y)
        xy = list(chain.from_iterable(zip(_x, _y)))
        self.vertex_list.vertices = xy

    def update_bounds(self,xmin,xmax,ymin,ymax):
        """
        Compute the bounds of this curves data and
        update the scale accordingly
        """
        self.xmin = min(self.xmin,xmin)
        self.ymin = min(self.ymin,ymin)
        self.xmax = max(self.xmax,xmax)
        self.ymax = max(self.ymax,ymax)

        self.scale.check_bounds(self.xmin,self.xmax,self.ymin,self.ymax)

def get_network_batch(x, y,
                      diameter,
                      draw_nodes_as_rectangles=False,
                      n_circle_segments=16,
                      node_color='#ffffff',
                      node_stroke_color='#111111',
                      ):
    """
    Create a batch for a network visualization.

    Parameters
    ----------
    x : numpy.ndarray(N, 2)
        positions
    LJ_r : float
        Lennard-Jones diameter
    n_circle_segments : bool, default = 16
        Number of segments a circle will be constructed of.

    Returns
    -------
    display_objects : dict
        A dictionary containing all the necessary objects to draw and
        update the network.

        - `disks` : a list of pyglet-circle objects (one entry per node)
        - `circles` : a list of pyglet-circle objects (one entry per node)
        - `batch` : the pyglet Batch instance that contains all of the objects
    """
    radius = diameter/2

    batch = pyglet.graphics.Batch()


    lines = []
    disks = []
    circles = []

    N = x.shape[0]

    draw_nodes = True
    if draw_nodes:
        disks = [None for n in range(N)]
        circles = [None for n in range(N)]


        for id, (_x, _y) in enumerate(zip(x,y)):
            if not draw_nodes_as_rectangles:
                disks[id] = \
                        shapes.Circle(_x,
                                      _y,
                                      radius,
                                      segments=n_circle_segments,
                                      color=tuple(bytes.fromhex(node_color[1:])),
                                      batch=batch,
                                      )

                circles[id] = \
                        shapes.Arc(_x,
                                   _y,
                                   radius,
                                   segments=n_circle_segments+1,
                                   color=tuple(bytes.fromhex(node_stroke_color[1:])),
                                   batch=batch,
                                   )
            else:
                disks[id] = \
                        shapes.Rectangle(
                                      _x,
                                      _y,
                                      2*radius,
                                      2*radius,
                                      color=tuple(bytes.fromhex(node_color[1:])),
                                      batch=batch)

    return {'disks': disks, 'circles':circles, 'batch': batch}

_default_config = {
            'plot_sampled_curve': True,
            'n_circle_segments':16,
            'plot_height':120,
            'bgcolor':'#eeeeee',
            'curve_stroke_width':4.0,
            'node_stroke_width':2.0,
            'link_color': '#4b5a62',
            'node_stroke_color':'#111111',
            'node_color':'#555555',
            'bound_increase_factor':1.0,
            'nodes_bound_increase_factor':1.2,
            'update_dt':0.04,
            'show_curves':True,
            'draw_nodes_as_rectangles':False,
            'show_legend': True,
            'legend_font_color':None,
            'legend_font_size':10,
            'padding':10,
            'compartment_colors':_colors,
            'palette': "dark",
        }

# light colors
#_default_config.update({
#            'bgcolor':'#fbfbef',
#            'link_color': '#8e9aaf',
#            'node_stroke_color':'#000000',
#            'legend_font_color':'#040414',
#        })


def visualize(simulation_kwargs,
              N_steps_per_frame,
              config=None,
              ignore_energies=[],
              width=400,
              height=400,
              simulation_api='py',
            ):
    """
    Start a visualization of a simulation.

    Parameters
    ==========
    simulation_kwargs : dict
        Keyword arguments of a simulation, will be fed to :func:`_ljhouses.simulate_once`.
    network: dict
        A stylized network in the netwulf-format
        (see https://netwulf.readthedocs.io/en/latest/python_api/post_back.html)
        where instead of 'x' and 'y', node positions are saved in 'x_canvas'
        and 'y_canvas'. Example:

        .. code:: python

            stylized_network = {
                "xlim": [0, 833],
                "ylim": [0, 833],
                "linkAlpha": 0.5,
                "nodeStrokeWidth": 0.75,
                "links": [
                    {"source": 0, "target": 1, "width": 3.0 }
                ],
                "nodes": [
                    {"id": 0,
                     "x_canvas": 436.0933431058901,
                     "y_canvas": 431.72418500564186,
                     "radius": 20
                     },
                    {"id": 1,
                     "x_canvas": 404.62184898400426,
                     "y_canvas": 394.8158724310507,
                     "radius": 20
                     }
                ]
            }

    N_steps_per_frame : float
        The amount of simulation time that's supposed to pass
        with a single update.
    ignore_energies : list, default = []
        List of compartment objects that are supposed to be
        ignored when plotted.
    quarantine_compartments : list, default = []
        List of compartment objects that are supposed to be
        resemble quarantine (i.e. temporarily
        losing all connections)
    config : dict, default = None
        A dictionary containing all possible configuration
        options. Entries in this dictionary will overwrite
        the default config which is

        .. code:: python

            _default_config = {
                        'plot_sampled_curve': True,
                        'draw_links':True,
                        'draw_nodes':True,
                        'n_circle_segments':16,
                        'plot_height':120,
                        'bgcolor':'#253237',
                        'curve_stroke_width':4.0,
                        'node_stroke_width':1.0,
                        'link_color': '#4b5a62',
                        'node_stroke_color':'#000000',
                        'node_color':'#264653',
                        'bound_increase_factor':1.0,
                        'update_dt':0.04,
                        'show_curves':True,
                        'draw_nodes_as_rectangles':False,
                        'show_legend': True,
                        'legend_font_color':'#fafaef',
                        'legend_font_size':10,
                        'padding':10,
                        'compartment_colors':_colors
                    }

    """

    x = simulation_kwargs['positions']
    v = simulation_kwargs['velocities']
    a = simulation_kwargs['accelerations']
    N = x.shape[0]
    LJ_r = simulation_kwargs['LJ_r']
    LJ_e = simulation_kwargs['LJ_e']
    LJ_Rmax = simulation_kwargs['LJ_Rmax']
    g = simulation_kwargs['g']
    dt = simulation_kwargs['dt']

    main_width = width
    main_height = height

    energies = ['K', 'V', 'Vij', 'E']

    if simulation_api == 'py':
        simulate = simulate_once_python
    else:
        simulate = simulate_once

    # update the config and compute some helper variables
    cfg = deepcopy(_default_config)
    if config is not None:
        cfg.update(config)

    #palette = cfg['palette']
    #if type(palette) == str:
    #    if 'link_color' not in cfg:
    #        cfg['link_color'] = col.hex_link_colors[palette]
    #    if 'bgcolor' not in cfg:
    #        cfg['bgcolor'] = col.hex_bg_colors[palette]
    #    if 'compartment_colors' not in cfg:
    #        cfg['compartment_colors'] = [ col.colors[this_color] for this_color in col.palettes[palette] ]

    cfg['compartment_colors'] = [ list(bytes.fromhex(c[1:])) for c in cfg['compartment_colors'] ]

    bgcolor = [ _/255 for _ in list(bytes.fromhex(cfg['bgcolor'][1:])) ] + [1.0]

    bgY = 0.2126*bgcolor[0] + 0.7152*bgcolor[1] + 0.0722*bgcolor[2]
    if cfg['legend_font_color'] is None:
        if bgY < 0.5:
            cfg['legend_font_color'] = '#fafaef'
        else:
            cfg['legend_font_color'] = '#232323'

    with_plot = cfg['show_curves']

    if with_plot:
        height += cfg['plot_height']
        plot_width = width
        plot_height = cfg['plot_height']
    else:
        plot_height = 0

    with_legend = cfg['show_legend']

    if with_legend:
        legend_batch = pyglet.graphics.Batch()
        #x, y = legend.get_location()
        #legend.set_location(x - width, y)
        # create a test label to get the actual dimensions
        test_label = pyglet.text.Label('Ag')
        dy = test_label.content_height * 1.1
        del(test_label)

        legend_circle_radius = dy/2/2
        distance_between_circle_and_label = 2*legend_circle_radius
        legend_height = len(energies) * dy + cfg['padding']

        # if legend is shown in concurrence to the plot,
        # move the legend to be on the right hand side of the plot,
        # accordingly make the plot at least as tall as 
        # the demanded height or the legend height
        if with_plot:
            plot_height = max(plot_height, legend_height)
        legend_y_offset = legend_height

        max_text_width = 0
        legend_objects = [] # this is a hack so that the garbage collector doesn't delete our stuff 
        for iC, C in enumerate(energies):
            this_y = legend_y_offset - iC * dy - cfg['padding']
            this_x = width + cfg['padding'] + legend_circle_radius
            label = pyglet.text.Label(str(C),
                              font_name=('Helvetica', 'Arial', 'Sans'),
                              font_size=cfg['legend_font_size'],
                              x=this_x + legend_circle_radius+distance_between_circle_and_label,
                              y=this_y,
                              anchor_x='left', anchor_y='top',
                              color = list(bytes.fromhex(cfg['legend_font_color'][1:])) + [255],
                              batch = legend_batch
                              )
            legend_objects.append(label)

            #if not cfg['draw_nodes_as_rectangles']:
            if True:
                disk = shapes.Circle(this_x,
                                      this_y - (dy-1.25*legend_circle_radius)/2,
                                      legend_circle_radius,
                                      segments=64,
                                      color=cfg['compartment_colors'][iC],
                                      batch=legend_batch,
                                      )

                circle = shapes.Arc(this_x,
                                      this_y - (dy-1.25*legend_circle_radius)/2,
                                      legend_circle_radius,
                                      segments=64+1,
                                      color=list(bytes.fromhex(cfg['legend_font_color'][1:])),
                                      batch=legend_batch,
                                      )

                legend_objects.extend([disk,circle])
            #else:
            #    rect = shapes.Rectangle(this_x,
            #                          this_y - (dy-1.5*legend_circle_radius)/2,
            #                          2*legend_circle_radius,
            #                          2*legend_circle_radius,
            #                          color = _colors[iC],
            #                          batch=legend_batch,
            #                          )
            #    legend_objects.append(rect)

            max_text_width = max(max_text_width, label.content_width)

        legend_width =   2*cfg['padding'] \
                       + 2*legend_circle_radius \
                       + distance_between_circle_and_label \
                       + max_text_width

        # if legend is shown in concurrence to the plot,
        # move the legend to be on the right hand side of the plot,
        # accordingly make the plot narrower and place the legend
        # directly under the square network plot.
        # if not, make the window wider and show the legend on
        # the right hand side of the network plot.
        if with_plot:
            for obj in legend_objects:
                obj.x -= legend_width
            plot_width = width - legend_width
        else:
            width += legend_width



    # overwrite network style with the epipack default style
    #network['linkColor'] = cfg['link_color']
    #network['nodeStrokeColor'] = cfg['node_stroke_color']
    #for node in network['nodes']:
    #    node['color'] = cfg['node_color']
    #N = len(network['nodes'])
    maxx = max(np.abs(x[:,0])) * 1.1
    maxy = max(np.abs(x[:,1])) * 1.1
    maxxx = max(maxx, maxy)

    main_scl = Scale(bound_increase_factor=cfg['nodes_bound_increase_factor'],is_square_domain=True)\
                        .extent(0,main_width,height,plot_height)\
                        .domain(-maxxx,maxxx,-maxxx,maxxx)
    winx, winy = main_scl.scale(x[:,0],x[:,1])
    win_diameter = main_scl.scale_dist(LJ_r)



    # get the OpenGL shape objects that comprise the network
    network_batch = get_network_batch(winx,winy,
                                      win_diameter,
                                      draw_nodes_as_rectangles=cfg['draw_nodes_as_rectangles'],
                                      n_circle_segments=cfg['n_circle_segments'],
                                      node_color=cfg['node_color'],
                                      node_stroke_color=cfg['node_stroke_color'],
                                      )
    disks = network_batch['disks']
    circles = network_batch['circles']
    batch = network_batch['batch']

    # initialize a simulation state that has to passed to the app
    # so the app can change simulation parameters
    simstate = SimulationStatus(N, N_steps_per_frame)
    simstate.update(x, v, a)

    # intialize app
    size = (width, height)
    window = App(*size,simulation_status=simstate,resizable=True)
    glClearColor(*bgcolor)

    # handle different strokewidths
    #if 'nodeStrokeWidth' in network:
    #    node_stroke_width = network['nodeStrokeWidth'] 
    #else:
    node_stroke_width = cfg['node_stroke_width']

    def _set_linewidth_nodes():
        glLineWidth(node_stroke_width)

    def _set_linewidth_curves():
        glLineWidth(cfg['curve_stroke_width'])

    def _set_linewidth_legend():
        glLineWidth(1.0)

    # add the network batch with the right function to set the linewidth
    # prior to drawing
    window.add_batch(batch,prefunc=_set_linewidth_nodes)

    if with_legend:
        # add the legend batch with the right function to set the linewidth
        # prior to drawing
        window.add_batch(legend_batch,prefunc=_set_linewidth_legend)

    # decide whether to plot all measured changes or only discrete-time samples
    discrete_plot = True

    # initialize time arrays

    t = 0.0
    time = [t]
    kinetic_energy = [_total_kinetic_energy(v)]
    potential_energy = [_total_potential_energy(x, g)]
    interaction_energy = [_total_interaction_energy(x, LJ_r, LJ_e, LJ_Rmax)]

    data = {
                'K': kinetic_energy,
                'V': potential_energy,
                'Vij': interaction_energy,
            }
    data['E'] = [sum([v[-1] for v in data.values()])]

    # initialize curves
    if with_plot:
        # find the maximal value of the
        # compartments that are meant to be plotted. 
        # These sets are needed for filtering later on.
        maxy = max([ data[C][-1] for C in (set(energies) - set(ignore_energies))])
        miny = min([ data[C][-1] for C in (set(energies) - set(ignore_energies))])
        scl = Scale(bound_increase_factor=cfg['bound_increase_factor'])\
                .extent(0,plot_width,plot_height-cfg['padding'],cfg['padding'])\
                .domain(0,20*N_steps_per_frame*dt,miny,maxy)
        curves = {}
        for iC, C in enumerate(energies):
            if C in ignore_energies:
                continue
            _batch = pyglet.graphics.Batch()
            window.add_batch(_batch,prefunc=_set_linewidth_curves)
            y = data[C]
            curve = Curve(time,y,cfg['compartment_colors'][iC],scl,_batch)
            curves[C] = curve

    # define the pyglet-App update function that's called on every clock cycle
    def update(dt):

        # skip if nothing remains to be done
        if simstate.simulation_ended or simstate.paused:
            return

        # get N_steps_per_frame
        N_steps_per_frame = simstate.N_steps_per_frame

        # Advance the simulation until time N_steps_per_frame.
        # sim_time is a numpy array including all time values at which
        # the system state changed. The first entry is the initial state
        # of the simulation at t = 0 which we will discard later on
        # the last entry at `N_steps_per_frame` will be missing so we
        # have to add it later on.
        # `sim_result` is a dictionary that maps a compartment
        # to a numpy array containing the compartment counts at
        # the corresponding time.
        simulation_kwargs['Nsteps'] = N_steps_per_frame
        simulation_kwargs['positions'] = simstate.x
        simulation_kwargs['velocities'] = simstate.v
        simulation_kwargs['accelerations'] = simstate.a
        x, v, a, K, V, Vij = simulate(**simulation_kwargs)


        time.append(time[-1] + N_steps_per_frame*dt)

        simstate.update(x, v, a)

        new_data = {'K': K, 'V': V, 'Vij': Vij, 'E': K+V+Vij}

        # compare the new node statuses with the old node statuses
        # and find the nodes that have changed status
        #ndx = np.where(model.node_status != simstate.old_node_status)[0]

        # if nothing changed, evaluate the true total event rate
        # and if it's zero, do not do anything anymore
        #did_simulation_end = len(ndx) == 0 and model.get_true_total_event_rate() == 0.0
        #simstate.set_simulation_status(did_simulation_end)
        if simstate.simulation_ended:
            return

        # advance the current time as described above.
        # we save both all time values as well as just the sampled times.

        # if curves are plotted
        if with_plot:

            # iterate through result array
            for k, v in new_data.items():
                # skip curves that should be ignored
                if k in ignore_energies:
                    continue
                curves[k].append_single_value(time[-1], v)

        pos = np.array(x)
        x = pos[:,0]
        y = pos[:,1]
        main_scl.check_square_bounds_centered_on_origin(x, y)
        _x, _y = main_scl.scale(x,y)
        R = main_scl.scale_dist(LJ_r) / 2
        for id, (__x, __y) in enumerate(zip(_x,_y)):
            disks[id].x = __x
            disks[id].y = __y
            disks[id].radius = R
            circles[id].x = __x
            circles[id].y = __y
            circles[id]._radius = R

            #if cfg['draw_nodes']:
            #    disks[node].color = cfg['compartment_colors'][status]


    # schedule the app clock and run the app
    pyglet.clock.schedule_interval(update, cfg['update_dt'])
    pyglet.app.run()
    #pyglet.clock.unschedule(update)
    #window.close()
    #del(window)
    #print(pyglet.app.platform_event_loop)
    #pyglet.app.exit()
    #print("called exit..")



if __name__=="__main__":     # pragma: no cover


    from ljhouses import simulation, StochasticBerendsenThermostat, NVEThermostat
    from ljhouses.tools import get_lattice_initial_conditions

    import numpy as np

    N = 2000
    LJ_r = 6
    LJ_e = 3
    LJ_Rmax = 3*LJ_r
    g = 0.3
    v0 = 10.0
    dt = 0.01

    x, v, a = get_lattice_initial_conditions(N, v0, LJ_r)
    thermostat = StochasticBerendsenThermostat(v0, N, berendsen_tau_as_multiple_of_dt=100)
    thermostat = NVEThermostat()
    simulation_kwargs = dict(
            positions = x,
            velocities = v,
            accelerations = a,
            LJ_r = LJ_r,
            LJ_e = LJ_e,
            LJ_Rmax = LJ_Rmax,
            g = g,
            dt = dt,
            thermostat = thermostat,
        )

    N_steps_per_frame = 10
    visualize(simulation_kwargs, N_steps_per_frame, width=800,height=800)
