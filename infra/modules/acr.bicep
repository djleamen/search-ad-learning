targetScope = 'resourceGroup'

param location string
param registryName string

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: registryName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
    publicNetworkAccess: 'Enabled'
  }
}

output registryName string = acr.name
output registryLoginServer string = acr.properties.loginServer
output registryId string = acr.id
