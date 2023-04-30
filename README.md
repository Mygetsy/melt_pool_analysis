# Melt Pool Size Analysis Tool For High Speed Video During Laser Surface Melting

## Description:
The tool was done to help in melt pool area detection on a set of low quality melt pool high speed videos with various light conditions and melt pool states.
The main propose is automatization of pre-processing of melt area detection and subsequent data analysis. 

## Installation
For instalation please check `requirements.txt`

## Possible functions 
The example of possible functions are provided in the self explanatory notebook.

``Melt pool area detection.ipynb``

  In order to run it, download sample [high speed video](https://disk.yandex.ru/d/XtKutCTDbCURzg) and place in the video folder.

## Example of results
<p align="center">
  <table>
    <tr>
      <td align="center">
        <figure>
          <img src="notebooks/Initial_image.gif" width="300" alt="Initial image">
          <figcaption> <br>Initial image </figcaption>
        </figure>
      </td>
      <td align="center">
        <figure>
          <img src="notebooks/Prospective_corrected_image.gif" width="300" alt="Prospective corrected image">
          <figcaption> <br>Prospective corrected image </figcaption>
        </figure>
      </td>
    </tr>
    <tr>
      <td align="center">
        <figure>
          <img src="notebooks/Points_on_the_border.gif" width="300" alt="Points on the border">
          <figcaption> <br>Points on the border </figcaption>
        </figure>
      </td>
      <td align="center">
        <figure>
          <img src="notebooks/Ellipse_approximation_of_the_melt_pool.gif" width="300" alt="Ellipse approximation of the melt pool">
          <figcaption> <br>Ellipse approximation of the melt pool </figcaption>
        </figure>
      </td>
    </tr>
  </table>
</p>

#### Data analysis

- Melt pool width, length and coordinate versus time
- Melt averaged intensity versus time


