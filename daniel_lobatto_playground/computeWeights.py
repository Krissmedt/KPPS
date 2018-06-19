import numpy as np

def computeWeights():
  nodes      = [0.0, 0.5, 1.0]
  half_nodes = [0.0, 0.25, 0.75]

  # First lagrange polynomial
  p1 = np.polyfit(nodes, [1.0, 0.0, 0.0], 2)
  p2 = np.polyfit(nodes, [0.0, 1.0, 0.0], 2)
  p3 = np.polyfit(nodes, [0.0, 0.0, 1.0], 2)

  intp1 = np.polyint(p1)
  intp2 = np.polyint(p2)
  intp3 = np.polyint(p3)

  q = np.zeros(3)
  q[0] = np.polyval(intp1,1.0)
  q[1] = np.polyval(intp2,1.0)
  q[2] = np.polyval(intp3,1.0)

  S = np.zeros((3,3))
  for j in range(3):
    if j==0:
      S[j,0] = np.polyval(intp1, nodes[j]) - np.polyval(intp1, 0.0)
      S[j,1] = np.polyval(intp2, nodes[j]) - np.polyval(intp2, 0.0)
      S[j,2] = np.polyval(intp3, nodes[j]) - np.polyval(intp3, 0.0)
    else:
      S[j,0] = np.polyval(intp1, nodes[j]) - np.polyval(intp1, nodes[j-1])
      S[j,1] = np.polyval(intp2, nodes[j]) - np.polyval(intp2, nodes[j-1])
      S[j,2] = np.polyval(intp3, nodes[j]) - np.polyval(intp3, nodes[j-1])

  S1 = np.zeros((3,3))
  for j in range(3): 
    if j==0:
      pass
    else:
      S1[j,0] = np.polyval(intp1, half_nodes[j]) - np.polyval(intp1, nodes[j-1])
      S1[j,1] = np.polyval(intp2, half_nodes[j]) - np.polyval(intp2, nodes[j-1])
      S1[j,2] = np.polyval(intp3, half_nodes[j]) - np.polyval(intp3, nodes[j-1])

  S2 = np.zeros((3,3))
  for j in range(3): 
    if j==0:
      pass
    else:
      S2[j,0] = np.polyval(intp1, nodes[j]) - np.polyval(intp1, half_nodes[j])
      S2[j,1] = np.polyval(intp2, nodes[j]) - np.polyval(intp2, half_nodes[j])
      S2[j,2] = np.polyval(intp3, nodes[j]) - np.polyval(intp3, half_nodes[j])

  # Generate a polynomial with random coefficient
  ptest    = np.random.rand(3)
  ptestval = np.polyval(ptest, nodes)
  ptestint = np.polyint(ptest)

  for j in range(3):

    if j==0:
      intexS = np.polyval(ptestint, nodes[0]) - np.polyval(ptestint, 0.0)
      assert np.abs( np.dot(S[0,:], ptestval) - intexS) < 1e-14, "Failed in S."
    else:
      intexS = np.polyval(ptestint, nodes[j]) - np.polyval(ptestint, nodes[j-1])
      assert np.abs( np.dot(S[j,:], ptestval) - intexS) < 1e-14, "Failed in S."

    #intexS1 = np.polyval(ptestint, half_nodes[j]) - np.polyval(ptestint, nodes[j])
    #assert np.abs( np.dot(S1[j,:], ptestval) - intexS1) < 1e-14, "Failed in S1."

    #intexS2 = np.polyval(ptestint, nodes[j]) - np.polyval(ptestint, half_nodes[j])
    #assert np.abs( np.dot(S2[j,:], ptestval) - intexS2) < 1e-14, "Failed in S."

  return S, S1, S2, q

